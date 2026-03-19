from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from greenpower_demo.config import QuantStrategyConfig
from greenpower_demo.quant_strategy import _apply_factor_scores, _build_backtest, _derive_factor_weights


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

    def test_backtest_produces_curves(self) -> None:
        config = QuantStrategyConfig(top_k=2, rebalance_frequency_days=2, holding_period_days=2)
        dates = pd.bdate_range("2024-01-01", periods=8)
        rows = []
        for idx, date in enumerate(dates):
            for symbol, score_bias in (("000537", 0.8), ("600795", 0.4), ("600011", 0.1)):
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "name": symbol,
                        "score": score_bias + idx * 0.01,
                        "rank": 1,
                        "limit_up_flag": 0.0,
                        "stock_return_1": 0.01 if symbol != "600011" else -0.005,
                    }
                )
        scored_frame = pd.DataFrame(rows)
        backtest_daily, rebalance_log, latest_portfolio = _build_backtest(
            scored_frame=scored_frame,
            test_start_date=pd.Timestamp(dates[1]),
            config=config,
        )
        self.assertFalse(backtest_daily.empty)
        self.assertFalse(rebalance_log.empty)
        self.assertEqual(len(latest_portfolio), 2)
        self.assertIn("strategy_curve", backtest_daily.columns)
        self.assertIn("benchmark_curve", backtest_daily.columns)

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


if __name__ == "__main__":
    unittest.main()
