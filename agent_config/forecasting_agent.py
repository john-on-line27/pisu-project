# agent_config/forecasting_agent.py
from google.adk import Agent
from .pipeline import forecasting


def generate_demand_forecast(steps: int = 14, test_size: int = 28) -> dict:
    """Trains a SARIMAX model on the cleaned demand time series, backtests
    it against the last `test_size` days (reporting MAE/RMSE/MAPE), and
    forecasts `steps` days ahead with an 80% confidence interval.

    Expects `outputs/demand_timeseries.csv` to already exist (run the data
    cleaning agent first).

    Args:
        steps: number of future days to forecast.
        test_size: number of most recent days held out for backtesting.

    Returns:
        A dict with backtest accuracy metrics, a preview of the forecast,
        and file paths for the full forecast CSV and chart.
    """
    return forecasting.train_backtest_forecast(steps=steps, test_size=test_size)


def generate_regional_demand_forecast(steps: int = 14) -> dict:
    """Forecasts demand separately for each Order Region and ranks them,
    both by full-history total demand (reliable) and by short-term SARIMAX
    forecast (only meaningful for regions with recent activity).

    Fits one SARIMAX model per region, which is slow (~20-30s for ~23
    regions), so this can be called more than once -- it saves progress to
    outputs/region_forecast.csv and resumes automatically. Check the
    `complete` field in the response; if False, call it again to continue.

    Expects `outputs/cleaned_orders.csv` to already exist (run the data
    cleaning agent first).

    Returns:
        A dict with `complete`, progress counts, and (once complete) two
        rankings: `ranked_by_historical_demand` and `ranked_by_forecast`.
        Each region also has a `data_coverage` flag ("active" or "sparse")
        -- sparse regions had little/no activity in the recent window, so
        their forecast reflects a data gap, not necessarily falling demand.
    """
    return forecasting.forecast_by_region(steps=steps)


def generate_item_demand_forecast(steps: int = 14, top_n: int = 15) -> dict:
    """Forecasts demand separately for the top `top_n` items (products, by
    historical volume) and ranks them, same approach as
    `generate_regional_demand_forecast` -- full-history total (reliable)
    plus a short-term SARIMAX forecast (only meaningful for items with
    recent activity).

    Only the top `top_n` items are modeled -- the long tail of products has
    too little volume for a meaningful per-item forecast. Saves/resumes
    progress at outputs/item_forecast.csv; check `complete` and call again
    if False.

    Expects `outputs/cleaned_orders.csv` to already exist (run the data
    cleaning agent first).

    Returns:
        A dict with `complete`, progress counts, and (once complete) two
        rankings: `ranked_by_historical_demand` and `ranked_by_forecast`.
        Each item also has a `data_coverage` flag ("active" or "sparse") --
        same caveat as regions: sparse items had little/no recent activity,
        so a low forecast there is a data gap, not necessarily falling
        demand.
    """
    return forecasting.forecast_by_item(steps=steps, top_n=top_n)


def generate_shipping_time_forecast(steps: int = 14, top_n: int = 15) -> dict:
    """Forecasts average shipping time (Days for shipping (real)) per
    destination country for the top `top_n` countries by order volume.

    Unlike the demand forecasts, this averages days rather than summing
    units, and fills gaps in the daily series by interpolation instead of
    zero (no orders that day doesn't mean zero shipping time). Same
    data_coverage caveat applies: countries with little/no recent activity
    are tagged "sparse" and their near-term forecast should be read as a
    data-gap artifact, not a real change in shipping performance -- prefer
    `historical_mean_days` for those.

    Saves/resumes progress at outputs/shipping_time_by_country.csv; check
    `complete` and call again if False.

    Expects `outputs/cleaned_orders.csv` to already exist (run the data
    cleaning agent first).

    Returns:
        A dict with `complete`, progress counts, and (once complete) two
        rankings: `ranked_by_historical_mean` and `ranked_by_forecast`.
    """
    return forecasting.forecast_shipping_time_by_country(steps=steps, top_n=top_n)


forecasting_agent = Agent(
    name="forecasting_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are the Demand Forecasting specialist for the PISU supply chain "
        "pipeline. Use `generate_demand_forecast` to train + backtest a SARIMAX "
        "model on the cleaned demand time series and produce a forward-looking "
        "forecast with confidence intervals. Use `generate_regional_demand_forecast` "
        "when asked which regions will have more/less demand, "
        "`generate_item_demand_forecast` when asked which items/products will have "
        "more/less demand, and `generate_shipping_time_forecast` when asked about "
        "shipping/delivery time by country -- these forecast each region/item/"
        "country separately and rank them. Always report the backtest error "
        "(MAE/RMSE/MAPE) alongside the overall forecast so the user knows how much "
        "to trust it. For regional, item, and shipping-time forecasts, always "
        "mention the `data_coverage` flag: entries marked 'sparse' had little/no "
        "recent activity in the source data, so their near-term forecast likely "
        "reflects a data gap rather than a real change -- prefer their historical "
        "ranking instead (or the analysis agent's `get_top_items_by_region` for a "
        "reliable historical view per region). Note that shipping time in this "
        "dataset is driven almost entirely by Shipping Mode (Same Day/First/"
        "Second/Standard Class), not destination country -- country-level "
        "differences are mostly noise. Flag if any forecasted values look "
        "unusually high (a potential demand spike worth a Slack/email alert per "
        "the project's n8n workflow). This agent requires the data cleaning agent "
        "to have run first."
    ),
    tools=[generate_demand_forecast, generate_regional_demand_forecast, generate_item_demand_forecast, generate_shipping_time_forecast],
)
