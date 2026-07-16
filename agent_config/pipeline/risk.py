# agent_config/pipeline/risk.py
"""Composite supply chain disruption risk score per region.

Combines several independent signals already produced by the cleaning,
analysis, and forecasting agents into one interpretable 0-100 score per
Order Region, so a business user gets a single ranked list of "where is
supply chain risk concentrated" instead of five separate spreadsheets.
"""
import os
import json
import pandas as pd
import numpy as np

# Equal weighting by default -- deliberately simple and auditable rather
# than fit/tuned, since there is no historical "did a disruption actually
# happen" label to validate a fitted model against in this dataset.
DEFAULT_WEIGHTS = {
    "late_delivery_rate": 0.25,
    "shipping_delay_severity": 0.20,
    "demand_volatility": 0.20,
    "data_coverage_gap": 0.20,
    "cancellation_fraud_rate": 0.15,
}


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def compute_disruption_risk_score(cleaned_path: str = "outputs/cleaned_orders.csv",
                                   status_path: str = "outputs/order_status_by_region.csv",
                                   region_col: str = "Order Region",
                                   recent_window: int = 90,
                                   weights: dict = None,
                                   save_dir: str = "outputs") -> dict:
    """Computes a 0-100 disruption risk score per region from five signals:

    1. late_delivery_rate - share of orders flagged Late_delivery_risk=1.
    2. shipping_delay_severity - mean (actual - scheduled) shipping days,
       floored at 0 (early shipments don't reduce risk).
    3. demand_volatility - coefficient of variation of daily order volume.
    4. data_coverage_gap - share of the final `recent_window` days with NO
       order activity at all. A region that has gone quiet is itself a risk
       signal (reporting gap, channel disruption, or real demand collapse
       -- all worth flagging even before knowing which).
    5. cancellation_fraud_rate - share of orders cancelled or flagged
       fraud, from outputs/order_status_by_region.csv (produced by the
       cleaning agent before those rows are dropped). If that file doesn't
       exist yet (pipeline run before this feature was added), this signal
       is skipped and weights are renormalized across the rest.

    Each raw signal is min-max normalized across regions (0=lowest risk in
    this dataset, 1=highest) before weighting, so the score is a relative
    ranking within this dataset, not an absolute probability.

    Returns a dict with the full ranked table and the path it was saved to.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    df = pd.read_csv(cleaned_path, parse_dates=["order_date"])
    full_dates = pd.date_range(df["order_date"].min().normalize(), df["order_date"].max().normalize(), freq="D")
    recent_cutoff = full_dates[-1] - pd.Timedelta(days=recent_window)

    rows = []
    for region, g in df.groupby(region_col):
        late_rate = float(g["Late_delivery_risk"].mean()) if "Late_delivery_risk" in g.columns else np.nan
        delay = (g["Days for shipping (real)"] - g["Days for shipment (scheduled)"]).clip(lower=0)
        delay_severity = float(delay.mean())

        daily = g.groupby(g["order_date"].dt.normalize())["Order Item Quantity"].sum()
        cv = float(daily.std() / daily.mean()) if daily.mean() else 0.0

        recent_days_with_orders = g[g["order_date"] > recent_cutoff]["order_date"].dt.normalize().nunique()
        coverage_gap = 1 - (recent_days_with_orders / recent_window)

        rows.append({
            region_col: region,
            "order_count": int(len(g)),
            "late_delivery_rate": round(late_rate, 4),
            "shipping_delay_severity_days": round(delay_severity, 3),
            "demand_volatility_cv": round(cv, 3),
            "data_coverage_gap": round(coverage_gap, 3),
        })
    result = pd.DataFrame(rows)

    # Optional 5th signal: cancellation/fraud rate, from the cleaning agent's
    # pre-filter snapshot.
    have_cancel_signal = os.path.exists(status_path)
    if have_cancel_signal:
        status = pd.read_csv(status_path)
        result = result.merge(status[[region_col, "cancelled_or_fraud_rate"]], on=region_col, how="left")
    else:
        weights.pop("cancellation_fraud_rate", None)

    # Renormalize weights to sum to 1 (in case a signal was skipped)
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    signal_cols = {
        "late_delivery_rate": "late_delivery_rate",
        "shipping_delay_severity": "shipping_delay_severity_days",
        "demand_volatility": "demand_volatility_cv",
        "data_coverage_gap": "data_coverage_gap",
        "cancellation_fraud_rate": "cancelled_or_fraud_rate",
    }
    score = pd.Series(0.0, index=result.index)
    for weight_key, w in weights.items():
        col = signal_cols[weight_key]
        score += w * _minmax(result[col])
    result["disruption_risk_score"] = (score * 100).round(1)

    result = result.sort_values("disruption_risk_score", ascending=False).reset_index(drop=True)
    result["risk_rank"] = result.index + 1

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "disruption_risk_by_region.csv")
    result.to_csv(out_path, index=False)

    weights_path = os.path.join(save_dir, "disruption_risk_weights.json")
    with open(weights_path, "w") as f:
        json.dump({"weights_used": weights, "cancellation_signal_available": have_cancel_signal}, f, indent=2)

    return {
        "results_path": out_path,
        "weights_used": weights,
        "cancellation_signal_available": have_cancel_signal,
        "ranked": result.to_dict(orient="records"),
    }
