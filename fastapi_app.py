"""
FastAPI service exposing the PISU forecasting pipeline over HTTP, so n8n
(or anything else) can call it on a schedule instead of running Python
directly. This is the missing middle piece of the README's roadmap:

[cleaning/analysis/forecasting agents] --> [FastAPI: /forecast] <-- [n8n]

Run it locally with:
    uvicorn fastapi_app:app --reload --port 8000

Or deploy it (e.g. to Railway) so n8n doesn't depend on a machine staying
on. Deployment note: `/forecast` re-runs the SARIMAX model live against
`outputs/demand_timeseries.csv` (small, ~40KB, safe to commit and fast to
run per-request). The other endpoints (`/forecast/region`, `/forecast/item`,
`/risk`, `/investment/*`, `/scenarios`) instead SERVE the pipeline's
precomputed result files directly rather than recomputing from
`outputs/cleaned_orders.csv` -- that raw file is ~90MB and deliberately
gitignored, so it won't exist in a deployed environment. Re-run the
pipeline locally (or add a scheduled job) and re-commit the small
outputs/*.csv and *.json files whenever you want the deployed API's
snapshot-based endpoints to refresh.

Then GET https://<your-deployment>/forecast?steps=14 returns the same
shape the existing get_sarimax_forecast() tool in agent_config/agent.py
already expects (point that tool's `url` at your deployed URL instead of
127.0.0.1 once deployed).
"""
import json
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from agent_config.pipeline import forecasting

app = FastAPI(
    title="PISU Supply Chain Forecast API",
    description="Serves demand, regional, item, risk, and investment forecasts from the PISU pipeline.",
    version="1.0.0",
)

OUTPUTS_DIR = "outputs"


def _read_csv_records(filename: str):
    """Reads a precomputed pipeline output CSV, replacing NaN (e.g. an
    unmeasurable growth % for a sparse-coverage item/category) with None
    so the response is valid JSON."""
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise HTTPException(
            status_code=503,
            detail=f"{path} not found. Run the pipeline locally and commit/redeploy this file.",
        )
    df = pd.read_csv(path)
    df = df.replace({np.nan: None})
    return df.to_dict(orient="records")


def _read_json(filename: str):
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise HTTPException(
            status_code=503,
            detail=f"{path} not found. Run the pipeline locally and commit/redeploy this file.",
        )
    with open(path) as f:
        return json.load(f)


@app.get("/")
def health():
    return {"status": "ok", "service": "pisu-forecast-api"}


@app.get("/forecast")
def get_forecast(steps: int = 14, test_size: int = 28, jitter: bool = True, jitter_pct: float = 0.20):
    """Aggregate demand forecast -- the endpoint the README's n8n workflow
    calls on a schedule. Returns the forecast series plus a single
    `max_forecast_value` field, which is what the n8n IF node should
    compare against its alert threshold.

    `jitter` (default True) applies random +/-jitter_pct noise to each
    forecast value before returning. The real SARIMAX model is
    deterministic given the same input data, so without this every call
    would return an identical result -- fine for a real deployment, but
    it means an n8n workflow calling this on a schedule would never see
    its IF-node threshold trip during testing/demo runs. This is a
    presentation-layer randomization for exercising both branches of the
    workflow, NOT a change to the underlying forecasting methodology --
    set jitter=false to get the model's real, unperturbed output.
    """
    try:
        result = forecasting.train_backtest_forecast(steps=steps, test_size=test_size)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="outputs/demand_timeseries.csv not found -- run the cleaning agent (run_pipeline.py) first.")

    fc_df = pd.read_csv(result["forecast_path"])

    if jitter:
        noise = np.random.uniform(1 - jitter_pct, 1 + jitter_pct, size=len(fc_df))
        fc_df["forecast"] = (fc_df["forecast"] * noise).round(1)
        fc_df["upper_80"] = (fc_df["upper_80"] * noise).round(1)
        fc_df["lower_80"] = (fc_df["lower_80"] * noise).clip(lower=0).round(1)

    peak = fc_df.loc[fc_df["forecast"].idxmax()]

    return {
        "steps": steps,
        "backtest_mape_pct": result["backtest_metrics"]["mape_pct"],
        "jitter_applied": jitter,
        "forecast": fc_df.to_dict(orient="records"),
        "max_forecast_value": float(peak["forecast"]),
        "max_forecast_date": str(peak["date"]),
    }


@app.get("/forecast/region")
def get_regional_forecast():
    """Per-region demand forecast, served from the pipeline's last precomputed
    outputs/region_forecast.csv snapshot (see agent_config/forecasting_agent.py
    to regenerate it locally)."""
    return {"results_path": "outputs/region_forecast.csv", "ranked": _read_csv_records("region_forecast.csv")}


@app.get("/forecast/item")
def get_item_forecast():
    """Per-item demand forecast, served from the precomputed
    outputs/item_forecast.csv snapshot."""
    return {"results_path": "outputs/item_forecast.csv", "ranked": _read_csv_records("item_forecast.csv")}


@app.get("/risk")
def get_risk():
    """Disruption risk score by region, served from the precomputed
    outputs/disruption_risk_by_region.csv snapshot."""
    weights_file = _read_json("disruption_risk_weights.json")
    return {
        "results_path": "outputs/disruption_risk_by_region.csv",
        "weights_used": weights_file.get("weights_used", weights_file),
        "cancellation_signal_available": weights_file.get("cancellation_signal_available"),
        "ranked": _read_csv_records("disruption_risk_by_region.csv"),
    }


@app.get("/investment/region")
def get_investment_region():
    """Risk-adjusted investment ranking by region, served from the precomputed
    outputs/investment_score_by_region.csv snapshot."""
    return {"results_path": "outputs/investment_score_by_region.csv", "ranked": _read_csv_records("investment_score_by_region.csv")}


@app.get("/investment/category")
def get_investment_category():
    """Risk-adjusted investment ranking by category, served from the precomputed
    outputs/investment_score_by_category.csv snapshot."""
    return {"results_path": "outputs/investment_score_by_category.csv", "ranked": _read_csv_records("investment_score_by_category.csv")}


@app.get("/scenarios")
def get_growth_scenarios():
    """12-month growth scenarios, served from the precomputed
    outputs/growth_scenarios.json snapshot."""
    return {"results_path": "outputs/growth_scenarios.json", "scenarios": _read_json("growth_scenarios.json")}
