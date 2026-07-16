# agent_config/pipeline/forecasting.py
"""Train, backtest, and forecast demand with SARIMAX."""
import os
import json
import warnings
import pandas as pd
import numpy as np

DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 7)  # weekly seasonality


def _fit(train: pd.Series, order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL_ORDER):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        return model.fit(disp=False)


def backtest(ts: pd.DataFrame, value_col: str = "units", test_size: int = 28,
             order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL_ORDER) -> dict:
    """Hold out the last `test_size` days, fit on the rest, and score."""
    series = ts.set_index("order_date")[value_col].asfreq("D").interpolate()
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]

    fitted = _fit(train, order, seasonal_order)
    pred = fitted.get_forecast(steps=test_size).predicted_mean
    pred.index = test.index

    mae = float(np.mean(np.abs(pred - test)))
    rmse = float(np.sqrt(np.mean((pred - test) ** 2)))
    nonzero = test.replace(0, np.nan)
    mape = float(np.mean(np.abs((pred - test) / nonzero)) * 100)

    return {
        "test_size_days": test_size,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "order": order,
        "seasonal_order": seasonal_order,
    }


def forecast(ts: pd.DataFrame, value_col: str = "units", steps: int = 14,
             order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL_ORDER) -> dict:
    """Fit SARIMAX on the full series and forecast `steps` days ahead with
    confidence intervals."""
    series = ts.set_index("order_date")[value_col].asfreq("D").interpolate()
    fitted = _fit(series, order, seasonal_order)

    fc = fitted.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.2)  # 80% CI

    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    result_df = pd.DataFrame({
        "date": future_dates,
        "forecast": mean.values.round(1),
        "lower_80": ci.iloc[:, 0].values.round(1),
        "upper_80": ci.iloc[:, 1].values.round(1),
    })
    return {"forecast_df": result_df, "model_aic": round(float(fitted.aic), 1)}


def train_backtest_forecast(ts_path: str = "outputs/demand_timeseries.csv",
                             value_col: str = "units", steps: int = 14,
                             test_size: int = 28, save_dir: str = "outputs") -> dict:
    """End-to-end: load timeseries -> backtest -> forecast -> save. Used by
    both the ADK tool wrapper and the standalone pipeline script."""
    ts = pd.read_csv(ts_path, parse_dates=["order_date"])
    # if timeseries has a group column (e.g. category), collapse to total demand
    if "Category Name" in ts.columns:
        ts = ts.groupby("order_date", as_index=False)[[value_col]].sum()

    bt = backtest(ts, value_col=value_col, test_size=test_size)
    fc = forecast(ts, value_col=value_col, steps=steps)

    os.makedirs(save_dir, exist_ok=True)
    fc_path = os.path.join(save_dir, "forecast_results.csv")
    fc["forecast_df"].to_csv(fc_path, index=False)

    metrics_path = os.path.join(save_dir, "forecast_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"backtest": bt, "model_aic": fc["model_aic"], "steps_forecasted": steps}, f, indent=2)

    # quick chart
    chart_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = ts.set_index("order_date")[value_col].asfreq("D").interpolate()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(hist.index[-90:], hist.values[-90:], label="Historical demand", color="#2b6cb0")
        ax.plot(fc["forecast_df"]["date"], fc["forecast_df"]["forecast"], label="Forecast", color="#e53e3e")
        ax.fill_between(fc["forecast_df"]["date"], fc["forecast_df"]["lower_80"], fc["forecast_df"]["upper_80"],
                         color="#e53e3e", alpha=0.15, label="80% CI")
        ax.set_title("Demand Forecast (SARIMAX)")
        ax.set_ylabel(value_col)
        ax.legend()
        fig.tight_layout()
        chart_path = os.path.join(save_dir, "forecast_chart.png")
        fig.savefig(chart_path, dpi=130)
        plt.close(fig)
    except Exception as e:
        chart_path = f"chart generation failed: {e}"

    return {
        "backtest_metrics": bt,
        "forecast_path": fc_path,
        "metrics_path": metrics_path,
        "chart_path": chart_path,
        "forecast_preview": fc["forecast_df"].head(7).to_dict(orient="records"),
    }


def forecast_by_region(cleaned_path: str = "outputs/cleaned_orders.csv",
                        region_col: str = "Order Region", steps: int = 14,
                        recent_window: int = 90, save_dir: str = "outputs",
                        time_budget_sec: float = 35.0) -> dict:
    """Forecast demand separately for each region and rank them.

    Fits one SARIMAX model per region (same order as the main model) on that
    region's own daily series, then reports both the region's full-history
    total (a reliable "which regions matter most overall" signal) and a
    short-term forecast (useful for the regions that have recent activity).

    IMPORTANT caveat this function surfaces explicitly: in the DataCo
    dataset, most regions have little or no order activity in the final
    `recent_window` days of the data's own timeline -- this looks like a
    data-coverage gap in how the dataset was compiled, not a real demand
    collapse. Regions with no recent activity get `data_coverage: "sparse"`
    and their forecast/growth numbers should be read with that caveat rather
    than taken as a real trend.

    Because fitting ~20+ SARIMAX models is slow, this function is safe to
    call repeatedly: it saves partial progress to `<save_dir>/region_forecast.csv`
    and resumes from there, so it can be run across multiple short calls if
    needed (see `time_budget_sec`).
    """
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    full_dates = pd.date_range(df["order_date"].min().normalize(), df["order_date"].max().normalize(), freq="D")
    recent_cutoff = full_dates[-1] - pd.Timedelta(days=recent_window)

    region_daily = (
        df.groupby([pd.Grouper(key="order_date", freq="D"), region_col])["Order Item Quantity"]
        .sum().reset_index()
    )
    regions_by_volume = list(df.groupby(region_col)["Order Item Quantity"].sum().sort_values(ascending=False).index)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "region_forecast.csv")
    done = {}
    if os.path.exists(out_path):
        done = {r["region"]: r for r in pd.read_csv(out_path).to_dict(orient="records")}

    import time
    start = time.time()
    for region in regions_by_volume:
        if region in done or time.time() - start > time_budget_sec:
            continue
        sub = region_daily[region_daily[region_col] == region].set_index("order_date")["Order Item Quantity"]
        sub = sub.reindex(full_dates, fill_value=0)
        recent = sub[sub.index > recent_cutoff]
        coverage = "sparse" if (recent > 0).sum() < recent_window * 0.2 else "active"

        try:
            fitted = _fit(sub)
            fc = fitted.get_forecast(steps=steps).predicted_mean.clip(lower=0)
            fc_total, fc_avg, status = float(fc.sum()), float(fc.mean()), "ok"
            recent_avg = float(recent.mean())
            growth_pct = round((fc_avg - recent_avg) / recent_avg * 100, 1) if recent_avg > 0 else None
        except Exception as e:
            fc_total = fc_avg = growth_pct = None
            status = f"failed: {str(e)[:80]}"
            recent_avg = float(recent.mean())

        done[region] = {
            "region": region,
            "historical_total_units": int(sub.sum()),
            "recent_window_avg_daily": round(recent_avg, 2),
            "data_coverage": coverage,
            "forecast_total_units": round(fc_total, 1) if fc_total is not None else None,
            "forecast_avg_daily": round(fc_avg, 2) if fc_avg is not None else None,
            "growth_vs_recent_pct": growth_pct,
            "status": status,
        }
        pd.DataFrame(list(done.values())).to_csv(out_path, index=False)

    result = pd.DataFrame(list(done.values()))
    complete = len(done) == len(regions_by_volume)
    return {
        "complete": complete,
        "regions_done": len(done),
        "regions_total": len(regions_by_volume),
        "results_path": out_path,
        "ranked_by_historical_demand": result.sort_values("historical_total_units", ascending=False).to_dict(orient="records") if complete else None,
        "ranked_by_forecast": result.sort_values("forecast_total_units", ascending=False).to_dict(orient="records") if complete else None,
    }


def forecast_by_item(cleaned_path: str = "outputs/cleaned_orders.csv",
                      item_col: str = "Product Name", top_n: int = 15, steps: int = 14,
                      recent_window: int = 90, save_dir: str = "outputs",
                      time_budget_sec: float = 35.0) -> dict:
    """Forecast demand separately for the top `top_n` items (by historical
    volume) and rank them. Same approach and same caveats as
    forecast_by_region: most items go quiet in the dataset's final
    `recent_window` days (a data-coverage gap, not real declining demand),
    so each item is tagged `data_coverage: "active"` or `"sparse"`.

    Only the top `top_n` items are modeled (of 118 distinct products in this
    dataset, the top ~15 account for the large majority of volume; the long
    tail is too sparse for a meaningful per-item SARIMAX fit anyway).

    Saves/resumes progress at `<save_dir>/item_forecast.csv`, same pattern
    as forecast_by_region, since fitting many models can exceed a single
    short call.
    """
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    full_dates = pd.date_range(df["order_date"].min().normalize(), df["order_date"].max().normalize(), freq="D")
    recent_cutoff = full_dates[-1] - pd.Timedelta(days=recent_window)

    item_daily = (
        df.groupby([pd.Grouper(key="order_date", freq="D"), item_col])["Order Item Quantity"]
        .sum().reset_index()
    )
    items_by_volume = list(df.groupby(item_col)["Order Item Quantity"].sum().sort_values(ascending=False).head(top_n).index)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "item_forecast.csv")
    done = {}
    if os.path.exists(out_path):
        done = {r["item"]: r for r in pd.read_csv(out_path).to_dict(orient="records")}

    import time
    start = time.time()
    for item in items_by_volume:
        if item in done or time.time() - start > time_budget_sec:
            continue
        sub = item_daily[item_daily[item_col] == item].set_index("order_date")["Order Item Quantity"]
        sub = sub.reindex(full_dates, fill_value=0)
        recent = sub[sub.index > recent_cutoff]
        coverage = "sparse" if (recent > 0).sum() < recent_window * 0.2 else "active"

        try:
            fitted = _fit(sub)
            fc = fitted.get_forecast(steps=steps).predicted_mean.clip(lower=0)
            fc_total, fc_avg, status = float(fc.sum()), float(fc.mean()), "ok"
            recent_avg = float(recent.mean())
            growth_pct = round((fc_avg - recent_avg) / recent_avg * 100, 1) if recent_avg > 0 else None
        except Exception as e:
            fc_total = fc_avg = growth_pct = None
            status = f"failed: {str(e)[:80]}"
            recent_avg = float(recent.mean())

        done[item] = {
            "item": item,
            "historical_total_units": int(sub.sum()),
            "recent_window_avg_daily": round(recent_avg, 2),
            "data_coverage": coverage,
            "forecast_total_units": round(fc_total, 1) if fc_total is not None else None,
            "forecast_avg_daily": round(fc_avg, 2) if fc_avg is not None else None,
            "growth_vs_recent_pct": growth_pct,
            "status": status,
        }
        pd.DataFrame(list(done.values())).to_csv(out_path, index=False)

    result = pd.DataFrame(list(done.values()))
    complete = len(done) == len(items_by_volume)
    return {
        "complete": complete,
        "items_done": len(done),
        "items_total": len(items_by_volume),
        "results_path": out_path,
        "ranked_by_historical_demand": result.sort_values("historical_total_units", ascending=False).to_dict(orient="records") if complete else None,
        "ranked_by_forecast": result.sort_values("forecast_total_units", ascending=False).to_dict(orient="records") if complete else None,
    }


def forecast_shipping_time_by_country(cleaned_path: str = "outputs/cleaned_orders.csv",
                                       country_col: str = "Order Country",
                                       value_col: str = "Days for shipping (real)",
                                       top_n: int = 15, steps: int = 14,
                                       recent_window: int = 90, save_dir: str = "outputs",
                                       time_budget_sec: float = 35.0) -> dict:
    """Forecasts average shipping time (days) per destination country for
    the top `top_n` countries by order volume (of ~164 countries in this
    dataset, most have too few orders for a meaningful daily average).

    Unlike the unit-based forecasts, this aggregates by mean (not sum) and
    fills gaps in the daily series by interpolation rather than zero --
    "no orders that day" doesn't mean "zero shipping time", so treating
    it as a demand series would bias the average down.

    Same data_coverage caveat as forecast_by_region/forecast_by_item: most
    countries have little/no order activity in the final `recent_window`
    days, so their near-term forecast reflects a data gap, not necessarily
    a real change in shipping performance -- prefer the historical mean for
    "sparse" countries.

    Saves/resumes progress at `<save_dir>/shipping_time_by_country.csv`.
    """
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    full_dates = pd.date_range(df["order_date"].min().normalize(), df["order_date"].max().normalize(), freq="D")
    recent_cutoff = full_dates[-1] - pd.Timedelta(days=recent_window)

    country_daily = (
        df.groupby([pd.Grouper(key="order_date", freq="D"), country_col])[value_col]
        .mean().reset_index()
    )
    countries_by_volume = list(df[country_col].value_counts().head(top_n).index)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "shipping_time_by_country.csv")
    done = {}
    if os.path.exists(out_path):
        done = {r["country"]: r for r in pd.read_csv(out_path).to_dict(orient="records")}

    import time
    start = time.time()
    for country in countries_by_volume:
        if country in done or time.time() - start > time_budget_sec:
            continue
        raw = df[df[country_col] == country]
        recent_days_with_orders = raw[raw["order_date"] > recent_cutoff]["order_date"].dt.normalize().nunique()
        coverage = "sparse" if recent_days_with_orders < recent_window * 0.2 else "active"

        sub = country_daily[country_daily[country_col] == country].set_index("order_date")[value_col]
        sub = sub.reindex(full_dates).interpolate().ffill().bfill()
        hist_mean = float(raw[value_col].mean())
        recent_mean = float(raw[raw["order_date"] > recent_cutoff][value_col].mean()) if recent_days_with_orders else None

        try:
            fitted = _fit(sub)
            fc = fitted.get_forecast(steps=steps).predicted_mean.clip(lower=0)
            fc_avg, status = float(fc.mean()), "ok"
        except Exception as e:
            fc_avg = None
            status = f"failed: {str(e)[:80]}"

        done[country] = {
            "country": country,
            "order_count": int(len(raw)),
            "historical_mean_days": round(hist_mean, 2),
            "recent_window_mean_days": round(recent_mean, 2) if recent_mean is not None else None,
            "data_coverage": coverage,
            "forecast_mean_days": round(fc_avg, 2) if fc_avg is not None else None,
            "status": status,
        }
        pd.DataFrame(list(done.values())).to_csv(out_path, index=False)

    result = pd.DataFrame(list(done.values()))
    complete = len(done) == len(countries_by_volume)
    return {
        "complete": complete,
        "countries_done": len(done),
        "countries_total": len(countries_by_volume),
        "results_path": out_path,
        "ranked_by_historical_mean": result.sort_values("historical_mean_days", ascending=False).to_dict(orient="records") if complete else None,
        "ranked_by_forecast": result.sort_values("forecast_mean_days", ascending=False).to_dict(orient="records") if complete else None,
    }
