# agent_config/investment_agent.py
from google.adk import Agent
from .pipeline import investment


def get_regional_investment_ranking() -> dict:
    """Ranks regions by a risk-adjusted investment score: historical
    profitability discounted by the disruption risk agent's risk score
    (a highly profitable but high-risk region gets discounted, not
    automatically ranked first).

    Expects `outputs/cleaned_orders.csv` to exist (run the data cleaning
    agent first). Best run after the risk agent, though it computes its
    own risk signals internally if needed.

    Returns:
        A dict with the full ranked table (`ranked`) and the file path
        results were saved to.
    """
    return investment.compute_investment_score_by_region()


def get_category_investment_ranking() -> dict:
    """Ranks product categories by the same risk-adjusted investment
    score methodology as `get_regional_investment_ranking`, plus recent-
    vs-historical revenue growth where a category has enough recent data
    to measure it (data_coverage: active) -- for "sparse" categories,
    growth is null and the score relies on historical profitability and
    risk alone.

    Expects `outputs/cleaned_orders.csv` to exist (run the data cleaning
    agent first).

    Returns:
        A dict with the full ranked table (`ranked`) and the file path
        results were saved to.
    """
    return investment.compute_investment_score_by_category()


def forecast_growth_scenarios(strategic_uplift: float = 0.05) -> dict:
    """Projects three explicit-assumption 12-month company revenue/profit
    scenarios (downside, base, upside) rather than extending the SARIMAX
    forecast further into a period the data can't reliably support.

    - downside: the data-coverage gap reflects a real contraction (only
      currently-active regions keep ordering).
    - base: historical 3-year steady state continues (the gap is a data
      artifact).
    - upside: base + measured growth in active regions, weighted by their
      revenue share, + a labeled `strategic_uplift` planning assumption
      (default 5%) for successfully executing the investment agent's
      recommendations.

    Expects `outputs/cleaned_orders.csv` and `outputs/region_forecast.csv`
    to exist (run the data cleaning and forecasting agents first).

    Args:
        strategic_uplift: assumed fractional revenue uplift (default 0.05
            = 5%) from successful execution of investment recommendations,
            layered on top of the data-derived upside growth.

    Returns:
        A dict with all three scenarios (`scenarios`), each including its
        annual revenue/profit projection, % vs. base, and a plain-language
        statement of the assumption driving it -- always report the
        assumption alongside the number, since these are scenario
        projections, not a single calibrated forecast.
    """
    return investment.compute_growth_scenarios(strategic_uplift=strategic_uplift)


investment_agent = Agent(
    name="investment_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are the Investment Strategy specialist for the PISU supply chain "
        "pipeline -- the synthesis agent that turns the cleaning, analysis, "
        "forecasting, and risk agents' outputs into a business recommendation. "
        "Use `get_regional_investment_ranking` and `get_category_investment_ranking` "
        "when asked where to invest to make the most money. Both combine "
        "historical profitability with a disruption risk discount, so always "
        "explain a top recommendation in terms of BOTH profit and risk (e.g. "
        "'high profit and low risk' vs 'high profit but discounted for risk'), "
        "not the score alone. For categories, also report the growth figure "
        "when available, and flag when a category's growth is unmeasurable "
        "(data_coverage: sparse) rather than omitting it silently. Use "
        "`forecast_growth_scenarios` when asked to predict overall company "
        "growth -- always present all three scenarios together with their "
        "stated assumptions, never a single number, since the honest answer "
        "to 'how much will the company grow' depends on whether the regional "
        "data gap is real. This agent requires the data cleaning agent to have "
        "run first, and is intended to run last, after analysis, forecasting, "
        "and risk."
    ),
    tools=[get_regional_investment_ranking, get_category_investment_ranking, forecast_growth_scenarios],
)
