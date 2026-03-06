from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from esg_alpha.config import PipelineConfig  # noqa: E402
from esg_alpha.factory import build_pipeline  # noqa: E402


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modular ESG alpha generation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--top", type=int, default=10, help="Show top N active weights.")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)
    pipeline = build_pipeline(config)
    result = pipeline.run()

    print("=" * 72)
    print("Pipeline complete")
    print("=" * 72)

    print("\nPerformance Metrics")
    percent_metrics = {
        "annualized_return",
        "annualized_volatility",
        "annualized_excess_return",
        "tracking_error",
        "max_drawdown",
        "cost_drag_annualized",
        "cumulative_return",
        "hit_rate",
    }

    for metric, value in sorted(result.evaluation.metrics.items()):
        if metric in percent_metrics:
            print(f"- {metric}: {_format_percent(value)}")
        else:
            print(f"- {metric}: {value:.4f}")

    latest_weights = result.target_weights.iloc[-1]
    active = latest_weights[latest_weights.abs() > 1e-6].sort_values(ascending=False)
    print("\nTop Active Weights")
    if active.empty:
        print("- No active positions on final date.")
    else:
        print(active.head(args.top).to_string())

    if not result.evaluation.factor_betas.empty:
        print("\nFactor Betas")
        print(result.evaluation.factor_betas.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
