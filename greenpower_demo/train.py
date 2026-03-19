from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from greenpower_demo.config import DEFAULT_START_DATE, DEFAULT_STRATEGY_ID, DEFAULT_SYMBOL, QuantStrategyConfig
    from greenpower_demo.quant_strategy import run_quant_strategy_pipeline
else:
    from .config import DEFAULT_START_DATE, DEFAULT_STRATEGY_ID, DEFAULT_SYMBOL, QuantStrategyConfig
    from .quant_strategy import run_quant_strategy_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the multi-factor quant strategy demo.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument("--strategy-id", default=DEFAULT_STRATEGY_ID)
    parser.add_argument("--holding-period-days", type=int, default=5)
    parser.add_argument("--rebalance-frequency-days", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--sell-tax-bps", type=float, default=5.0)
    parser.add_argument("--force-refresh", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = QuantStrategyConfig(
        strategy_id=args.strategy_id,
        holding_period_days=args.holding_period_days,
        rebalance_frequency_days=args.rebalance_frequency_days,
        top_k=args.top_k,
        train_ratio=args.train_ratio,
        transaction_cost_bps=args.transaction_cost_bps,
        sell_tax_bps=args.sell_tax_bps,
    )
    result = run_quant_strategy_pipeline(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        force_refresh=args.force_refresh,
        config=config,
    )
    summary = result["run_summary"]
    metrics = result["metrics"]
    print(
        json.dumps(
            {
                "strategy_id": summary["strategy_id"],
                "symbol": summary["symbol"],
                "latest_trade_date": summary["latest_trade_date"],
                "train_end_date": summary["train_end_date"],
                "holding_period_days": summary["holding_period_days"],
                "rebalance_frequency_days": summary["rebalance_frequency_days"],
                "top_k": summary["top_k"],
                "latest_holdings": summary["latest_holdings"],
                "metrics": metrics,
                "artifact_dir": result["artifact_dir"],
                "site_data_path": result["site_data_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
