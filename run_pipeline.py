#!/usr/bin/env python3
"""Run the clean -> analyze -> forecast pipeline directly, without going
through the ADK/Gemini agent loop (no GOOGLE_API_KEY required).

The exact same functions here (agent_config/pipeline/*.py) back the ADK
tools in data_cleaning_agent.py, analysis_agent.py, and forecasting_agent.py,
so `python main.py` (chat with the agents) and this script produce
consistent results.
"""
import json
from agent_config.pipeline import cleaning, analysis, forecasting


def main():
    print("=" * 60)
    print("STAGE 1/3: Data Cleaning Agent")
    print("=" * 60)
    clean_result = cleaning.clean_and_aggregate(freq="D")
    print(json.dumps(clean_result["report"], indent=2))
    print(f"Cleaned orders  -> {clean_result['cleaned_path']}")
    print(f"Demand timeseries -> {clean_result['timeseries_path']} "
          f"({clean_result['timeseries_rows']} days, "
          f"{clean_result['date_range'][0]} to {clean_result['date_range'][1]})")

    print("\n" + "=" * 60)
    print("STAGE 2/3: Analysis Agent")
    print("=" * 60)
    analysis_result = analysis.run_analysis()
    print(json.dumps(analysis_result, indent=2, default=str))

    print("\n" + "=" * 60)
    print("STAGE 3/3: Forecasting Agent")
    print("=" * 60)
    fc_result = forecasting.train_backtest_forecast(steps=14, test_size=28)
    print("Backtest metrics:", json.dumps(fc_result["backtest_metrics"], indent=2))
    print(f"Forecast saved -> {fc_result['forecast_path']}")
    print(f"Chart saved -> {fc_result['chart_path']}")
    print("Next 7 days preview:")
    for row in fc_result["forecast_preview"]:
        print(f"  {row}")

    print("\nDone. All outputs in ./outputs/")


if __name__ == "__main__":
    main()
