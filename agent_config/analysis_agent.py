# agent_config/analysis_agent.py
from google.adk import Agent
from .pipeline import analysis


def analyze_demand_timeseries() -> dict:
    """Runs descriptive and statistical analysis on the cleaned demand time
    series produced by the data cleaning agent: summary statistics, trend
    direction/strength, weekly seasonality strength, and top categories/
    regions by units and revenue.

    Expects `outputs/cleaned_orders.csv` and `outputs/demand_timeseries.csv`
    to already exist (run the data cleaning agent first).

    Returns:
        A dict with `summary`, `trend_seasonality`, and `breakdowns`, and
        the path the full report was saved to.
    """
    return analysis.run_analysis()


def get_top_items_by_region(top_n: int = 3) -> dict:
    """For each region, returns the top `top_n` items (products) by
    historical units sold -- e.g. "what sells where". This is descriptive
    (no forecasting model), so it's reliable for every region even though
    most regions have little recent activity in this dataset (see the
    forecasting agent's regional/item forecasts for the near-term view,
    which only really works for regions/items with recent data).

    Expects `outputs/cleaned_orders.csv` to already exist (run the data
    cleaning agent first).
    """
    return analysis.top_items_by_region(top_n=top_n)


analysis_agent = Agent(
    name="analysis_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are the Data Analysis specialist for the PISU supply chain pipeline. "
        "Use `analyze_demand_timeseries` to compute summary statistics, trend and "
        "weekly seasonality strength, and top categories/regions from the cleaned "
        "demand time series. Use `get_top_items_by_region` when asked what sells "
        "best in each region -- it's a reliable historical ranking, unlike the "
        "forecasting agent's per-item/per-region forecasts which only work well "
        "for regions/items with recent data. Explain findings in plain language "
        "(e.g. 'demand is trending up ~X units/day' or 'strong weekly seasonality, "
        "likely weekend effect'). This agent requires the data cleaning agent to "
        "have run first. Hand off to the forecasting agent once analysis is done."
    ),
    tools=[analyze_demand_timeseries, get_top_items_by_region],
)
