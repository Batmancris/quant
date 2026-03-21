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
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--initial-train-days", type=int, default=756)
    parser.add_argument("--walk-forward-test-days", type=int, default=63)
    parser.add_argument("--walk-forward-step-days", type=int, default=63)
    parser.add_argument("--universe-size", type=int, default=72)
    parser.add_argument("--universe-per-industry-cap", type=int, default=4)
    parser.add_argument("--max-positions-per-industry", type=int, default=2)
    parser.add_argument("--max-single-weight", type=float, default=0.18)
    parser.add_argument("--max-turnover", type=float, default=0.60)
    parser.add_argument("--min-market-cap-quantile", type=float, default=0.20)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--sell-tax-bps", type=float, default=5.0)
    parser.add_argument("--initial-cash", type=float, default=1000000.0)
    parser.add_argument("--broker-mode", default="paper")
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
        initial_train_days=args.initial_train_days,
        walk_forward_test_days=args.walk_forward_test_days,
        walk_forward_step_days=args.walk_forward_step_days,
        universe_size=args.universe_size,
        universe_per_industry_cap=args.universe_per_industry_cap,
        max_positions_per_industry=args.max_positions_per_industry,
        max_single_weight=args.max_single_weight,
        max_turnover=args.max_turnover,
        min_market_cap_quantile=args.min_market_cap_quantile,
        transaction_cost_bps=args.transaction_cost_bps,
        sell_tax_bps=args.sell_tax_bps,
        initial_cash=args.initial_cash,
        broker_mode=args.broker_mode,
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
                "success_rate": summary["success_rate"],
                "metrics": metrics,
                "simulation_account": result["simulation_account"],
                "trade_plan": result["trade_plan"],
                "artifact_dir": result["artifact_dir"],
                "site_data_path": result["site_data_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
