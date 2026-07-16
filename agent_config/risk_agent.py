# agent_config/risk_agent.py
from google.adk import Agent
from .pipeline import risk


def compute_disruption_risk_score(recent_window: int = 90) -> dict:
    """Computes a composite 0-100 supply chain disruption risk score per
    Order Region, combining five signals: late-delivery rate, shipping
    delay severity, demand volatility, data-coverage gap (region gone
    quiet recently), and cancellation/fraud rate.

    Each signal is min-max normalized across regions and combined with
    transparent, equal-ish weights (see agent_config/pipeline/risk.py) --
    this is a relative ranking within the dataset, not a calibrated
    probability, since there's no historical "disruption actually
    happened" label to fit against.

    Expects `outputs/cleaned_orders.csv` to already exist (run the data
    cleaning agent first). If `outputs/order_status_by_region.csv` also
    exists (produced by a recent cleaning-agent run), the cancellation/
    fraud signal is included; otherwise it's skipped and weights are
    renormalized across the remaining four signals.

    Returns:
        A dict with the full ranked table (`ranked`), the weights actually
        used, and the file path results were saved to.
    """
    return risk.compute_disruption_risk_score(recent_window=recent_window)


risk_agent = Agent(
    name="risk_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are the Supply Chain Risk specialist for the PISU pipeline. Use "
        "`compute_disruption_risk_score` to rank regions by a composite "
        "disruption risk score combining late deliveries, shipping delays, "
        "demand volatility, data-coverage gaps, and cancellation/fraud rate. "
        "Always name which signals drove a region's high score (don't just "
        "report the number) -- e.g. a region can be high-risk because of late "
        "deliveries, or because it has gone quiet in the data (a coverage gap, "
        "which may be a data artifact rather than a real disruption -- cross-"
        "check against the forecasting agent's data_coverage flag before "
        "recommending action). This agent requires the data cleaning agent to "
        "have run first, and should typically run after the analysis and "
        "forecasting agents so its findings can be cross-checked against theirs."
    ),
    tools=[compute_disruption_risk_score],
)
