# agent_config/agent.py
import os
import requests
import pandas as pd
import kagglehub
from google.adk import Agent

from .data_cleaning_agent import data_cleaning_agent
from .analysis_agent import analysis_agent
from .forecasting_agent import forecasting_agent
from .risk_agent import risk_agent
from .investment_agent import investment_agent


# Define tool 1: Call your FastAPI backend (n8n pipeline target)
def get_sarimax_forecast(steps: int = 7) -> dict:
    """
    Calls the local FastAPI endpoint to generate future supply chain volume forecasts
    using the trained SARIMAX model.

    Args:
        steps: Number of future intervals (days/weeks) to predict. Default is 7.
    """
    # Defaults to the local FastAPI server; set FORECAST_API_URL in .env to
    # point this at a deployed instance instead (e.g. your Railway URL +
    # "/forecast"), so this tool isn't dependent on 127.0.0.1 staying up.
    url = os.environ.get("FORECAST_API_URL", "http://127.0.0.1:8000/forecast")
    try:
        response = requests.get(url, params={"steps": steps}, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": f"FastAPI returned status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to FastAPI endpoint: {str(e)}"}


# Define tool 2: Inspect raw DataCo metadata locally
def query_supply_chain_metadata() -> dict:
    """
    Inspects the local DataCo dataset file downloaded from Kaggle to return
    the list of columns and high-risk late delivery categories.
    """
    try:
        path = kagglehub.dataset_download("saicharankomati/dataco-supply-chain-dataset")
        csv_name = os.listdir(path)[0]
        df = pd.read_csv(os.path.join(path, csv_name), encoding='latin-1', nrows=1000)

        # Pull quick metrics
        top_categories = df['Category Name'].value_counts().head(3).to_dict()
        top_regions = df['Order Region'].value_counts().head(3).to_dict()

        return {
            "columns": df.columns.tolist()[:10],  # Show first 10 columns
            "total_rows_scanned": len(df),
            "top_categories": top_categories,
            "top_regions": top_regions
        }
    except Exception as e:
        return {"error": f"Failed to inspect data: {str(e)}"}


# Define the root ADK Agent. It orchestrates the three pipeline specialists
# (data_cleaning_agent -> analysis_agent -> forecasting_agent) as sub-agents,
# transferring control to whichever stage the user's request needs.
root_agent = Agent(
    name="pisu_supply_chain_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are the PISU Supply Chain Assistant, orchestrating a 3-stage demand "
        "forecasting pipeline over the DataCo dataset:\n"
        "1. data_cleaning_agent - loads the raw export, cleans it, and aggregates "
        "it into a daily demand time series. Always run this first.\n"
        "2. analysis_agent - computes summary stats, trend, and seasonality on the "
        "cleaned time series. Run after cleaning.\n"
        "3. forecasting_agent - trains/backtests SARIMAX and produces a forward "
        "demand forecast with confidence intervals, at aggregate, regional, item, "
        "and shipping-time granularity. Run after cleaning (analysis is optional "
        "but recommended first).\n"
        "4. risk_agent - scores each region's supply chain disruption risk "
        "(late deliveries, shipping delays, demand volatility, data-coverage "
        "gaps, cancellation/fraud rate) into one ranked composite score. Run "
        "after cleaning; best run after forecasting so its findings (e.g. a "
        "region going quiet) can be cross-checked against the forecasting "
        "agent's data_coverage flags.\n"
        "5. investment_agent - the synthesis stage: combines historical "
        "profitability with the risk agent's risk score into a risk-adjusted "
        "investment ranking, for both regions and product categories. Run "
        "last, after risk.\n\n"
        "When the user asks to clean data, analyze it, forecast demand, assess "
        "disruption risk, or find the best regions/categories to invest in, "
        "transfer to the matching sub-agent. If they ask for 'the full pipeline' "
        "from scratch, run the sub-agents in order: cleaning -> analysis -> "
        "forecasting -> risk -> investment. Use `get_sarimax_forecast` to query an "
        "already-deployed FastAPI forecast endpoint, and `query_supply_chain_metadata` "
        "to inspect the raw Kaggle dataset directly. If a forecast exceeds safety "
        "thresholds, or a region's disruption risk score is notably high, recommend "
        "flagging it for an n8n Slack/Email alert."
    ),
    tools=[get_sarimax_forecast, query_supply_chain_metadata],
    sub_agents=[data_cleaning_agent, analysis_agent, forecasting_agent, risk_agent, investment_agent],
)
