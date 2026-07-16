# agent_config/pipeline/analysis.py
"""Descriptive/statistical analysis of the cleaned demand time series:
summary stats, trend & seasonality, category/region breakdowns.
"""
import os
import pandas as pd
import numpy as np


def summary_stats(ts: pd.DataFrame, value_col: str = "units") -> dict:
    s = ts[value_col]
    return {
        "n_periods": int(len(ts)),
        "mean": round(float(s.mean()), 2),
        "std": round(float(s.std()), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "total": round(float(s.sum()), 2),
        "coefficient_of_variation": round(float(s.std() / s.mean()), 3) if s.mean() else None,
    }


def trend_and_seasonality(ts: pd.DataFrame, value_col: str = "units", period: int = 7) -> dict:
    """Classical seasonal decomposition (additive) to quantify trend
    direction/strength and weekly seasonality strength."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    series = ts.set_index("order_date")[value_col].asfreq("D").interpolate()
    decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")

    trend = decomp.trend.dropna()
    trend_slope = float(np.polyfit(range(len(trend)), trend.values, 1)[0])
    resid_var = float(np.nanvar(decomp.resid))
    detrended_var = float(np.nanvar(series - decomp.trend))
    seasonal_strength = max(0.0, 1 - resid_var / detrended_var) if detrended_var else 0.0

    return {
        "trend_slope_per_day": round(trend_slope, 3),
        "trend_direction": "increasing" if trend_slope > 0.5 else ("decreasing" if trend_slope < -0.5 else "flat"),
        "weekly_seasonality_strength_0to1": round(seasonal_strength, 3),
        "period_used_days": period,
    }


def top_breakdowns(df: pd.DataFrame, n: int = 5) -> dict:
    """Top categories/regions by units and revenue, from the cleaned
    order-item level data (pre-aggregation)."""
    out = {}
    if "Category Name" in df.columns:
        out["top_categories_by_units"] = (
            df.groupby("Category Name")["Order Item Quantity"].sum().sort_values(ascending=False).head(n).round(0).to_dict()
        )
        out["top_categories_by_revenue"] = (
            df.groupby("Category Name")["Sales"].sum().sort_values(ascending=False).head(n).round(2).to_dict()
        )
    if "Order Region" in df.columns:
        out["top_regions_by_units"] = (
            df.groupby("Order Region")["Order Item Quantity"].sum().sort_values(ascending=False).head(n).round(0).to_dict()
        )
    return out


def top_items_by_region(cleaned_path: str = "outputs/cleaned_orders.csv",
                         item_col: str = "Product Name", region_col: str = "Order Region",
                         top_n: int = 3, save_dir: str = "outputs") -> dict:
    """For each region, the top `top_n` items by historical units sold.

    This is descriptive (no model fitting), so unlike the SARIMAX-based
    forecasts it works reliably for every region regardless of how much
    recent data that region has -- it answers "what sells where" over the
    full history rather than "what will sell next", which is the more
    trustworthy question to ask of this dataset given how thin the data
    gets in its final months (see forecast_by_region / forecast_by_item).
    """
    df = pd.read_csv(cleaned_path)
    grouped = (
        df.groupby([region_col, item_col])["Order Item Quantity"]
        .sum().reset_index().rename(columns={"Order Item Quantity": "units"})
    )
    region_totals = df.groupby(region_col)["Order Item Quantity"].sum().sort_values(ascending=False)

    result = {}
    for region in region_totals.index:
        top = (grouped[grouped[region_col] == region]
               .sort_values("units", ascending=False).head(top_n))
        result[region] = {
            "region_total_units": int(region_totals[region]),
            "top_items": [{"item": r[item_col], "units": int(r["units"])} for _, r in top.iterrows()],
        }

    os.makedirs(save_dir, exist_ok=True)
    import json
    out_path = os.path.join(save_dir, "top_items_by_region.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return {"regions": result, "saved_to": out_path}


def run_analysis(cleaned_path: str = "outputs/cleaned_orders.csv",
                  ts_path: str = "outputs/demand_timeseries.csv",
                  save_dir: str = "outputs") -> dict:
    """End-to-end analysis used by both the ADK tool wrapper and the
    standalone pipeline script."""
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    ts = pd.read_csv(ts_path, parse_dates=["order_date"])

    result = {
        "summary": summary_stats(ts),
        "trend_seasonality": trend_and_seasonality(ts),
        "breakdowns": top_breakdowns(df),
    }

    os.makedirs(save_dir, exist_ok=True)
    import json
    out_path = os.path.join(save_dir, "analysis_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    result["saved_to"] = out_path
    return result
