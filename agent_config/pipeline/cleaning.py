# agent_config/pipeline/cleaning.py
"""Load and clean the raw DataCo supply-chain export, then aggregate it
into a daily demand time series suitable for SARIMAX.
"""
import os
import pandas as pd
import numpy as np

RAW_DATA_CANDIDATES = [
    os.path.join("data", "dataco_supply_chain_sample.csv"),
    os.path.join("data", "dataco_supply_chain.csv"),
]

CANCEL_STATUSES = {"CANCELED", "SUSPECTED_FRAUD"}


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw DataCo CSV.

    If `path` isn't given, tries the local sample file first, falling back
    to a kagglehub download (requires network + kaggle credentials).
    """
    if path is None:
        for candidate in RAW_DATA_CANDIDATES:
            if os.path.exists(candidate):
                path = candidate
                break
    if path is None:
        import kagglehub
        dl_path = kagglehub.dataset_download("saicharankomati/dataco-supply-chain-dataset")
        path = os.path.join(dl_path, os.listdir(dl_path)[0])

    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    return df


def clean_data(df: pd.DataFrame) -> dict:
    """Clean the raw order-item dataframe.

    Handles: duplicate rows, malformed/mixed date formats, inconsistent
    category casing, negative/invalid quantities, missing sales values,
    and cancelled/fraudulent orders (excluded from demand).

    Returns a dict with the cleaned dataframe plus a report of what was
    fixed, so a caller (human or agent) can see exactly what changed.
    """
    report = {"input_rows": len(df)}
    out = df.copy()

    # 1. Drop exact duplicate rows
    before = len(out)
    out = out.drop_duplicates()
    report["duplicates_removed"] = before - len(out)

    # 2. Normalize category casing (title case)
    if "Category Name" in out.columns:
        out["Category Name"] = out["Category Name"].astype(str).str.strip().str.title()

    # 3. Parse order date, tolerating mixed formats
    date_col = "order date (DateOrders)"
    if date_col in out.columns:
        parsed = pd.to_datetime(out[date_col], errors="coerce", format="mixed")
        report["unparseable_dates_dropped"] = int(parsed.isna().sum())
        out["order_date"] = parsed
        out = out[out["order_date"].notna()]

    # 4. Fix invalid quantities (negative -> absolute value; treat as data-entry sign error)
    qty_col = "Order Item Quantity"
    if qty_col in out.columns:
        report["negative_quantities_fixed"] = int((out[qty_col] < 0).sum())
        out[qty_col] = out[qty_col].abs()
        report["zero_or_null_quantities_dropped"] = int((out[qty_col].isna() | (out[qty_col] == 0)).sum())
        out = out[out[qty_col].notna() & (out[qty_col] > 0)]

    # 5. Fill missing Sales from price * quantity * (1 - discount) where possible, else drop
    if {"Sales", "Order Item Product Price", qty_col}.issubset(out.columns):
        missing_sales = out["Sales"].isna()
        report["missing_sales_imputed"] = int(missing_sales.sum())
        disc = out.get("Order Item Discount Rate", 0)
        imputed = out[qty_col] * out["Order Item Product Price"] * (1 - disc.fillna(0))
        out.loc[missing_sales, "Sales"] = imputed[missing_sales]
        out = out[out["Sales"].notna()]

    # 6. Capture per-region order status mix BEFORE dropping cancelled/fraud rows --
    # this is the only point in the pipeline where that signal is still visible,
    # and it feeds the disruption risk score (agent_config/pipeline/risk.py).
    status_by_region = None
    if {"Order Status", "Order Region"}.issubset(out.columns):
        status_by_region = (
            out.groupby("Order Region")["Order Status"]
            .apply(lambda s: pd.Series({
                "total_orders": len(s),
                "cancelled_or_fraud": int(s.isin(CANCEL_STATUSES).sum()),
                "cancelled_or_fraud_rate": round(float(s.isin(CANCEL_STATUSES).mean()), 4),
            }))
            .unstack()
            .reset_index()
        )

    # 7. Drop cancelled / fraudulent orders -> not real demand
    if "Order Status" in out.columns:
        before = len(out)
        out = out[~out["Order Status"].isin(CANCEL_STATUSES)]
        report["cancelled_fraud_rows_dropped"] = before - len(out)

    report["output_rows"] = len(out)
    report["rows_removed_total"] = report["input_rows"] - report["output_rows"]
    return {"data": out, "report": report, "status_by_region": status_by_region}


def aggregate_to_timeseries(df: pd.DataFrame, freq: str = "D", group_col: str | None = None) -> pd.DataFrame:
    """Aggregate cleaned order-item rows into a demand time series.

    Args:
        df: cleaned dataframe (must have order_date, Order Item Quantity, Sales)
        freq: pandas offset alias, e.g. "D" (daily) or "W" (weekly)
        group_col: optional column to also group by (e.g. "Category Name")
    """
    keys = [pd.Grouper(key="order_date", freq=freq)]
    if group_col:
        keys.append(group_col)

    agg = (
        df.groupby(keys)
        .agg(
            units=("Order Item Quantity", "sum"),
            revenue=("Sales", "sum"),
            orders=("Order Id", "nunique") if "Order Id" in df.columns else ("Sales", "count"),
        )
        .reset_index()
        .sort_values("order_date")
    )
    return agg


def clean_and_aggregate(raw_path: str | None = None, freq: str = "D",
                         group_col: str | None = None, save_dir: str = "outputs") -> dict:
    """End-to-end: load -> clean -> aggregate -> save. Used by both the ADK
    tool wrapper and the standalone pipeline script."""
    raw = load_raw_data(raw_path)
    cleaned = clean_data(raw)
    ts = aggregate_to_timeseries(cleaned["data"], freq=freq, group_col=group_col)

    os.makedirs(save_dir, exist_ok=True)
    cleaned_path = os.path.join(save_dir, "cleaned_orders.csv")
    ts_path = os.path.join(save_dir, "demand_timeseries.csv")
    cleaned["data"].to_csv(cleaned_path, index=False)
    ts.to_csv(ts_path, index=False)

    status_path = None
    if cleaned.get("status_by_region") is not None:
        status_path = os.path.join(save_dir, "order_status_by_region.csv")
        cleaned["status_by_region"].to_csv(status_path, index=False)

    return {
        "report": cleaned["report"],
        "cleaned_path": cleaned_path,
        "timeseries_path": ts_path,
        "status_by_region_path": status_path,
        "timeseries_rows": len(ts),
        "date_range": [str(ts["order_date"].min()), str(ts["order_date"].max())],
    }
