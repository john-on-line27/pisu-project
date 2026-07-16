# agent_config/pipeline/investment.py
"""Risk-adjusted investment opportunity scoring: combines historical
profitability with disruption risk (regions) or profitability, recent
growth, and risk (categories) into one ranked recommendation.

This is deliberately the "last mile" of the pipeline: cleaning, analysis,
forecasting, and risk scoring each answer a narrower question; this module
synthesizes their outputs into the question a business decision-maker
actually asks -- "where should we put money to make the most of it".
"""
import os
import pandas as pd
import numpy as np


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def _risk_signals(df: pd.DataFrame, group_col: str, recent_window: int = 90) -> pd.DataFrame:
    """Shared 4-signal risk computation (late delivery, shipping delay,
    demand volatility, data-coverage gap), reused for both regions and
    categories so the two rankings are methodologically consistent."""
    full_dates = pd.date_range(df["order_date"].min().normalize(), df["order_date"].max().normalize(), freq="D")
    recent_cutoff = full_dates[-1] - pd.Timedelta(days=recent_window)

    rows = []
    for key, g in df.groupby(group_col):
        late_rate = float(g["Late_delivery_risk"].mean()) if "Late_delivery_risk" in g.columns else 0.0
        delay = (g["Days for shipping (real)"] - g["Days for shipment (scheduled)"]).clip(lower=0)
        delay_severity = float(delay.mean())
        daily = g.groupby(g["order_date"].dt.normalize())["Sales"].sum()
        cv = float(daily.std() / daily.mean()) if daily.mean() else 0.0
        recent = g[g["order_date"] > recent_cutoff]
        recent_days_with_orders = recent["order_date"].dt.normalize().nunique()
        coverage_gap = 1 - (recent_days_with_orders / recent_window)
        rows.append({
            group_col: key,
            "late_delivery_rate": late_rate,
            "shipping_delay_severity_days": delay_severity,
            "demand_volatility_cv": cv,
            "data_coverage_gap": coverage_gap,
        })
    risk = pd.DataFrame(rows)
    score = (
        0.30 * _minmax(risk["late_delivery_rate"])
        + 0.25 * _minmax(risk["shipping_delay_severity_days"])
        + 0.20 * _minmax(risk["demand_volatility_cv"])
        + 0.25 * _minmax(risk["data_coverage_gap"])
    )
    risk["risk_score"] = (score * 100).round(1)
    return risk


def compute_investment_score_by_region(cleaned_path: str = "outputs/cleaned_orders.csv",
                                        region_col: str = "Order Region",
                                        save_dir: str = "outputs") -> dict:
    """Ranks regions by a risk-adjusted investment score: historical
    profitability (total Order Profit Per Order) discounted by disruption
    risk. investment_score = profit_score * (1 - risk_score/100) * 100,
    so a highly profitable but high-risk region is discounted rather than
    automatically ranked first -- the same logic as a Sharpe-ratio-style
    risk-adjusted return, adapted to the signals available in this dataset.
    """
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])

    profit = df.groupby(region_col).agg(
        total_profit=("Order Profit Per Order", "sum"),
        avg_profit_per_order=("Order Profit Per Order", "mean"),
        total_revenue=("Sales", "sum"),
        order_count=("Order Profit Per Order", "size"),
    ).reset_index()

    risk = _risk_signals(df, region_col)
    result = profit.merge(risk[[region_col, "risk_score"]], on=region_col)

    result["profit_score_0to1"] = _minmax(result["total_profit"])
    result["investment_score"] = (result["profit_score_0to1"] * (1 - result["risk_score"] / 100) * 100).round(1)
    result = result.sort_values("investment_score", ascending=False).reset_index(drop=True)
    result["investment_rank"] = result.index + 1

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "investment_score_by_region.csv")
    result.to_csv(out_path, index=False)
    return {"results_path": out_path, "ranked": result.to_dict(orient="records")}


def compute_investment_score_by_category(cleaned_path: str = "outputs/cleaned_orders.csv",
                                          category_col: str = "Category Name",
                                          recent_window: int = 90,
                                          save_dir: str = "outputs") -> dict:
    """Ranks product categories by a risk-adjusted investment score,
    the category-level analogue of compute_investment_score_by_region,
    additionally reporting recent-vs-historical revenue growth where the
    category has enough recent activity to measure it (see data_coverage)."""
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    full_dates = pd.date_range(df["order_date"].min().normalize(), df["order_date"].max().normalize(), freq="D")
    recent_cutoff = full_dates[-1] - pd.Timedelta(days=recent_window)

    profit = df.groupby(category_col).agg(
        total_profit=("Order Profit Per Order", "sum"),
        avg_profit_per_order=("Order Profit Per Order", "mean"),
        total_revenue=("Sales", "sum"),
        order_count=("Order Profit Per Order", "size"),
    ).reset_index()

    risk = _risk_signals(df, category_col, recent_window)
    result = profit.merge(risk[[category_col, "risk_score", "data_coverage_gap"]], on=category_col)
    result["data_coverage"] = np.where(result["data_coverage_gap"] < 0.8, "active", "sparse")

    # Recent vs. historical revenue growth, only meaningful where coverage is active
    growth_rows = []
    for key, g in df.groupby(category_col):
        recent = g[g["order_date"] > recent_cutoff]
        recent_days = recent["order_date"].dt.normalize().nunique()
        if recent_days >= recent_window * 0.2:
            recent_avg_daily = recent["Sales"].sum() / recent_days
            hist_avg_daily = g["Sales"].sum() / g["order_date"].dt.normalize().nunique()
            growth = round((recent_avg_daily - hist_avg_daily) / hist_avg_daily * 100, 1) if hist_avg_daily else None
        else:
            growth = None
        growth_rows.append({category_col: key, "revenue_growth_recent_vs_hist_pct": growth})
    result = result.merge(pd.DataFrame(growth_rows), on=category_col)

    result["profit_score_0to1"] = _minmax(result["total_profit"])
    result["investment_score"] = (result["profit_score_0to1"] * (1 - result["risk_score"] / 100) * 100).round(1)
    result = result.sort_values("investment_score", ascending=False).reset_index(drop=True)
    result["investment_rank"] = result.index + 1
    result = result.drop(columns=["data_coverage_gap"])

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "investment_score_by_category.csv")
    result.to_csv(out_path, index=False)
    return {"results_path": out_path, "ranked": result.to_dict(orient="records")}


def compute_growth_scenarios(cleaned_path: str = "outputs/cleaned_orders.csv",
                              region_forecast_path: str = "outputs/region_forecast.csv",
                              strategic_uplift: float = 0.05,
                              recent_window: int = 90,
                              save_dir: str = "outputs") -> dict:
    """Projects three 12-month company revenue/profit scenarios from the
    cleaned data and the regional forecast agent's growth figures. This is
    deliberately NOT a longer-horizon SARIMAX extrapolation -- extending a
    model built on a dataset with a known coverage gap further into the
    future would just extrapolate that gap. Instead each scenario states
    an explicit business assumption and computes its revenue/profit
    consequence directly from real historical figures:

    - Downside ("data gap reflects a real contraction"): if the regions
      that went quiet in the final `recent_window` days genuinely stopped
      ordering (rather than a data-coverage artifact), annualized revenue
      is the current active-region-only daily run rate, extrapolated.
    - Base ("historical steady state / gap is a data artifact"): the full
      3-year historical daily average continues unchanged -- the
      assumption if the coverage gap is confirmed to be a reporting
      artifact and operations continue as before across all regions.
    - Upside ("execute the investment agent's recommendations"): base
      plus (a) the measured growth rate in the regions the forecasting
      agent flagged as active, weighted by their share of total revenue,
      plus (b) an explicit, labeled `strategic_uplift` assumption
      representing management successfully redirecting investment toward
      the top-ranked regions/categories (default 5%, NOT derived from the
      model -- a stated planning assumption, distinct from the other two
      scenarios which are computed directly from data).

    All three scenarios use the dataset's overall profit margin
    (total profit / total revenue) throughout, as a simplifying
    assumption -- margin is not modeled to vary by scenario.
    """
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    n_days = (df["order_date"].max() - df["order_date"].min()).days + 1
    n_years = n_days / 365.25

    total_revenue = float(df["Sales"].sum())
    total_profit = float(df["Order Profit Per Order"].sum())
    margin = total_profit / total_revenue
    base_revenue = total_revenue / n_years
    base_profit = total_profit / n_years

    # Downside: active-region-only run rate (see forecast_by_region for
    # which regions are "active" vs "sparse")
    region_fc = pd.read_csv(region_forecast_path)
    active_regions = region_fc[region_fc["data_coverage"] == "active"]["region"].tolist()
    cutoff = df["order_date"].max() - pd.Timedelta(days=recent_window)
    recent = df[df["order_date"] > cutoff]
    active_recent = recent[recent["Order Region"].isin(active_regions)]
    active_recent_days = active_recent["order_date"].dt.normalize().nunique()
    downside_daily_revenue = active_recent["Sales"].sum() / active_recent_days if active_recent_days else 0
    downside_revenue = downside_daily_revenue * 365.25
    downside_profit = downside_revenue * margin

    # Upside: base + revenue-share-weighted active-region growth + strategic uplift
    apac_share = df[df["Order Region"].isin(active_regions)]["Sales"].sum() / total_revenue
    active_growth = region_fc[region_fc["data_coverage"] == "active"]["growth_vs_recent_pct"]
    avg_active_growth = float(active_growth.mean()) / 100 if len(active_growth) else 0.0
    blended_growth = apac_share * avg_active_growth
    total_upside_growth = blended_growth + strategic_uplift
    upside_revenue = base_revenue * (1 + total_upside_growth)
    upside_profit = base_profit * (1 + total_upside_growth)

    scenarios = {
        "downside": {
            "label": "Downside -- data gap reflects a real contraction",
            "annual_revenue": round(downside_revenue, 0),
            "annual_profit": round(downside_profit, 0),
            "vs_base_pct": round((downside_revenue / base_revenue - 1) * 100, 1),
            "assumption": f"Only currently-active regions ({', '.join(active_regions)}) continue ordering; all others remain at zero.",
        },
        "base": {
            "label": "Base -- historical steady state continues",
            "annual_revenue": round(base_revenue, 0),
            "annual_profit": round(base_profit, 0),
            "vs_base_pct": 0.0,
            "assumption": "The 3-year historical daily average continues unchanged; the coverage gap is a data artifact, not a real change in ordering behavior.",
        },
        "upside": {
            "label": "Upside -- execute investment agent recommendations",
            "annual_revenue": round(upside_revenue, 0),
            "annual_profit": round(upside_profit, 0),
            "vs_base_pct": round((upside_revenue / base_revenue - 1) * 100, 1),
            "assumption": (
                f"Base, plus the measured growth rate in active regions ({avg_active_growth*100:.1f}% avg) "
                f"weighted by their {apac_share*100:.1f}% revenue share, plus a {strategic_uplift*100:.0f}% "
                "management-execution uplift from redirecting investment toward top-ranked regions/categories "
                "(a stated planning assumption, not a statistical forecast)."
            ),
        },
        "margin_used": round(margin, 4),
    }

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "growth_scenarios.json")
    import json
    with open(out_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    return {"results_path": out_path, "scenarios": scenarios}
