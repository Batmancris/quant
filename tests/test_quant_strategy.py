from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from greenpower_demo.config import QuantStrategyConfig
from greenpower_demo.execution import rebalance_paper_account
from greenpower_demo.quant_strategy import (
    _apply_factor_scores,
    _apply_turnover_limit,
    _build_backtest,
    _derive_factor_weights,
    _select_portfolio,
)


class QuantStrategyTests(unittest.TestCase):
    def test_factor_weights_normalize(self) -> None:
        ic_history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
                "factor": ["relative_strength_5", "relative_strength_5", "low_volatility", "low_volatility"],
                "ic": [0.10, 0.20, -0.05, -0.10],
            }
        )
        weights = _derive_factor_weights(ic_history)
        self.assertAlmostEqual(float(weights["weight"].abs().sum()), 1.0)
        self.assertIn("factor_label", weights.columns)

    def test_apply_factor_scores_adds_rank(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "symbol": ["000537", "600795"],
                "relative_strength_5_z": [1.0, -1.0],
                "trend_strength_z": [0.5, -0.5],
            }
        )
        factor_weights = pd.DataFrame(
            {
                "factor": ["relative_strength_5", "trend_strength"],
                "weight": [0.7, 0.3],
            }
        )
        scored = _apply_factor_scores(frame, factor_weights)
        self.assertIn("score", scored.columns)
        best = scored.sort_values("score", ascending=False).iloc[0]["symbol"]
        self.assertEqual(best, "000537")

    def test_turnover_limit_blends_weights(self) -> None:
        target = {"000537": 1.0}
        previous = {"600795": 1.0}
        adjusted, turnover, alpha = _apply_turnover_limit(target, previous, max_turnover=0.5)
        self.assertGreater(turnover, 0.5)
        self.assertLess(alpha, 1.0)
        self.assertAlmostEqual(sum(adjusted.values()), 1.0, places=6)

    def test_select_portfolio_respects_industry_cap(self) -> None:
        config = QuantStrategyConfig(top_k=3, max_positions_per_industry=1)
        group = pd.DataFrame(
            {
                "symbol": ["000537", "600795", "600519", "601318"],
                "name": ["绿发电力", "国电电力", "贵州茅台", "中国平安"],
                "industry": ["公用事业", "公用事业", "食品饮料", "非银金融"],
                "score": [0.9, 0.8, 0.7, 0.6],
                "market_cap_est": [100, 90, 120, 110],
                "market_cap_rank": [0.8, 0.7, 0.9, 0.85],
                "limit_up_flag": [0.0, 0.0, 0.0, 0.0],
            }
        )
        portfolio = _select_portfolio(group, config=config, previous_weights={})
        self.assertLessEqual(portfolio["industry"].value_counts().max(), 1)
        self.assertLessEqual(len(portfolio), 3)

    def test_backtest_produces_curves(self) -> None:
        config = QuantStrategyConfig(top_k=2, rebalance_frequency_days=2, holding_period_days=2)
        dates = pd.bdate_range("2024-01-01", periods=8)
        rows = []
        for idx, trade_date in enumerate(dates):
            for symbol, name, industry, score_bias in (
                ("000537", "绿发电力", "公用事业", 0.8),
                ("600795", "国电电力", "公用事业", 0.4),
                ("600519", "贵州茅台", "食品饮料", 0.2),
            ):
                rows.append(
                    {
                        "date": trade_date,
                        "symbol": symbol,
                        "name": name,
                        "industry": industry,
                        "score": score_bias + idx * 0.01,
                        "rank": 1,
                        "limit_up_flag": 0.0,
                        "market_cap_rank": 0.8,
                        "market_cap_est": 100,
                        "stock_return_1": 0.01 if symbol != "600519" else -0.005,
                    }
                )
        scored_frame = pd.DataFrame(rows)
        backtest_daily, rebalance_log, _ = _build_backtest(scored_frame=scored_frame, test_dates=list(dates[1:]), config=config)
        self.assertFalse(backtest_daily.empty)
        self.assertFalse(rebalance_log.empty)
        self.assertIn("strategy_curve", backtest_daily.columns)
        self.assertIn("benchmark_curve", backtest_daily.columns)

    def test_rebalance_paper_account_generates_orders(self) -> None:
        latest_portfolio = pd.DataFrame(
            {
                "symbol": ["000537", "600519"],
                "target_weight": [0.6, 0.4],
            }
        )
        price_frame = pd.DataFrame(
            {
                "symbol": ["000537", "600519"],
                "close": [10.0, 100.0],
            }
        )
        account_path = Path("tests") / "paper_account_test.json"
        try:
            _, orders = rebalance_paper_account(
                account_path=account_path,
                latest_portfolio=latest_portfolio,
                price_frame=price_frame,
                initial_cash=100000.0,
                lot_size=100,
            )
            self.assertTrue(account_path.exists())
            self.assertGreater(len(orders), 0)
            loaded = json.loads(account_path.read_text(encoding="utf-8"))
            self.assertIn("positions", loaded)
            self.assertGreater(loaded["equity"], 0)
        finally:
            if account_path.exists():
                account_path.unlink()


if __name__ == "__main__":
    unittest.main()
