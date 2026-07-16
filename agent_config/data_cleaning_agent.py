# agent_config/data_cleaning_agent.py
from google.adk import Agent
from .pipeline import cleaning


def clean_and_aggregate_data(freq: str = "D", group_by_category: bool = False) -> dict:
    """Loads the raw DataCo order-item export, cleans it (dedupes, fixes
    bad dates/quantities, imputes missing sales, drops cancelled/fraud
    orders), and aggregates it into a demand time series.

    Args:
        freq: aggregation frequency, "D" for daily or "W" for weekly.
        group_by_category: if True, also breaks the time series out by
            product Category Name (useful for per-category forecasts).

    Returns:
        A dict with the cleaning report (rows removed and why) and the
        file paths of the cleaned data and resulting time series.
    """
    group_col = "Category Name" if group_by_category else None
    return cleaning.clean_and_aggregate(freq=freq, group_col=group_col)


data_cleaning_agent = Agent(
    name="data_cleaning_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are the Data Cleaning specialist for the PISU supply chain pipeline. "
        "Use `clean_and_aggregate_data` to load the raw DataCo order-item export, "
        "clean it, and aggregate it into a daily (or weekly) demand time series. "
        "Always report what was removed/fixed (duplicates, bad dates, negative "
        "quantities, missing sales, cancelled/fraud orders) so the user can trust "
        "the output. Hand off to the analysis agent once cleaning is done."
    ),
    tools=[clean_and_aggregate_data],
)
