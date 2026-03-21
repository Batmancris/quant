from __future__ import annotations

import json
import random
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_START_DATE,
    DEFAULT_STRATEGY_ID,
    DEFAULT_SYMBOL,
    DISPLAY_NAME,
    QuantStrategyConfig,
    TrainingConfig,
    artifact_dir,
    feature_cache_path,
)
from .data import fetch_market_data
from .execution import build_ai_trade_context, rebalance_paper_account
from .features import build_features
from .site_export import export_static_site

FACTOR_SPECS: list[tuple[str, str]] = [
    ("relative_strength_5", "相对强弱"),
    ("trend_strength", "趋势强度"),
    ("low_volatility", "低波动"),
    ("turnover_stability", "换手稳定"),
    ("roe_quality", "ROE 质量"),
    ("cashflow_quality_factor", "现金流质量"),
    ("gross_margin_factor", "毛利率质量"),
    ("dividend_factor", "股息率"),
    ("value_factor", "估值分位"),
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if std <= 1e-12 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def _load_or_build_feature_frame(
    symbol: str,
    start_date: str,
    end_date: str | None,
    force_refresh: bool,
    config: QuantStrategyConfig,
) -> pd.DataFrame:
    cache_symbol = f"{symbol}_multi_asset_quant_h{config.holding_period_days}_u{config.universe_size}"
    cache_path = feature_cache_path(cache_symbol, start_date, end_date)
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, parse_dates=["date", "report_date", "fundamental_asof_date", "dividend_announce_date"], dtype={"symbol": str})

    market_data = fetch_market_data(symbol=symbol, start=start_date, end=end_date, force_refresh=force_refresh, config=config)
    build_config = TrainingConfig(prediction_horizon_days=config.holding_period_days, return_threshold=0.0, verbose=False)
    metadata_map = market_data.universe_metadata.set_index("symbol").to_dict(orient="index")

    frames: list[pd.DataFrame] = []
    for stock_symbol, stock_frame in market_data.stocks.items():
        meta = metadata_map.get(stock_symbol, {"name": stock_symbol, "industry": "未知行业"})
        frame, _ = build_features(
            stock_frame,
            market_data.indices,
            market_data.industry_proxy,
            market_data.northbound,
            config=build_config,
            symbol=stock_symbol,
            industry_fallback=str(meta.get("industry", "未知行业")),
            fundamental_history=market_data.fundamental_history,
            dividend_history=market_data.dividend_history,
        )
        if len(frame) >= 120:
            frame["name"] = meta.get("name", stock_symbol)
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    combined.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return combined


def _prepare_factor_frame(feature_frame: pd.DataFrame, config: QuantStrategyConfig) -> tuple[pd.DataFrame, list[str]]:
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)

    universe_proxy = frame[["date", "universe_index_close"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    universe_proxy["future_universe_return"] = universe_proxy["universe_index_close"].shift(-config.holding_period_days) / universe_proxy["universe_index_close"] - 1
    frame = frame.merge(universe_proxy[["date", "future_universe_return"]], on="date", how="left")

    industry_future = (
        frame.groupby(["date", "industry"], as_index=False)["future_return_horizon"]
        .mean()
        .rename(columns={"future_return_horizon": "future_industry_return"})
    )
    frame = frame.merge(industry_future, on=["date", "industry"], how="left")
    frame["future_excess_return"] = frame["future_return_horizon"] - frame["future_industry_return"]

    frame["market_cap_rank"] = frame.groupby("date")["market_cap_est"].rank(pct=True, method="average")
    frame["pb_rank"] = frame.groupby("date")["pb_ratio"].rank(pct=True, method="average")
    frame["pe_rank"] = frame.groupby("date")["pe_ratio"].rank(pct=True, method="average")

    frame["relative_strength_5"] = frame["stock_return_5"] - frame["universe_return_5"]
    frame["trend_strength"] = frame["ma_20_dev"]
    frame["low_volatility"] = -frame["volatility_20"]
    frame["turnover_stability"] = -frame["stock_turnover_change_1"].abs()
    frame["roe_quality"] = frame["roe"] / 100.0
    frame["cashflow_quality_factor"] = frame["cashflow_quality"]
    frame["gross_margin_factor"] = frame["gross_margin"] / 100.0
    frame["dividend_factor"] = frame["dividend_yield"]
    frame["value_factor"] = 1.0 - frame[["pb_rank", "pe_rank"]].mean(axis=1)

    factor_columns = [factor_name for factor_name, _ in FACTOR_SPECS]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.loc[frame["market_cap_rank"].fillna(0.0) >= config.min_market_cap_quantile].copy()

    for factor_name in factor_columns:
        industry_z = frame.groupby(["date", "industry"])[factor_name].transform(_zscore)
        date_z = frame.groupby("date")[factor_name].transform(_zscore)
        frame[f"{factor_name}_z"] = industry_z.where(industry_z.notna(), date_z).fillna(0.0)

    return frame.reset_index(drop=True), factor_columns


def _compute_factor_ic_history(frame: pd.DataFrame, factor_columns: list[str], min_cross_section: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = frame.dropna(subset=["future_excess_return"])
    for trade_date, group in base.groupby("date", sort=True):
        if len(group) < min_cross_section:
            continue
        target = group["future_excess_return"]
        if target.nunique() < 2:
            continue
        for factor_name in factor_columns:
            ic = group[f"{factor_name}_z"].corr(target, method="spearman")
            if pd.notna(ic):
                rows.append({"date": trade_date, "factor": factor_name, "ic": float(ic)})

    history = pd.DataFrame(rows)
    if history.empty:
        raise ValueError("无法计算因子 IC 历史，横截面样本不足。")

    history = history.sort_values(["factor", "date"]).reset_index(drop=True)
    history["cum_mean_ic"] = history.groupby("factor")["ic"].transform(lambda values: values.expanding().mean())
    return history


def _derive_factor_weights(ic_history: pd.DataFrame) -> pd.DataFrame:
    summary = (
        ic_history.groupby("factor", as_index=False)["ic"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_ic", "std": "ic_std", "count": "obs_count"})
    )
    summary["ic_ir"] = summary["mean_ic"] / summary["ic_std"].replace(0, np.nan)
    summary["raw_weight"] = summary["ic_ir"].fillna(summary["mean_ic"]).fillna(0.0)
    summary["raw_weight"] = summary["raw_weight"].where(summary["mean_ic"] > 0, 0.0)
    if float(summary["raw_weight"].abs().sum()) <= 1e-12:
        summary["raw_weight"] = summary["mean_ic"].abs().replace(0, np.nan).fillna(1.0)
    summary["weight"] = summary["raw_weight"] / summary["raw_weight"].abs().sum()
    summary["factor_label"] = summary["factor"].map(dict(FACTOR_SPECS))
    return summary.sort_values("weight", ascending=False).reset_index(drop=True)


def _apply_factor_scores(frame: pd.DataFrame, factor_weights: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = 0.0
    for row in factor_weights.itertuples(index=False):
        scored["score"] += float(row.weight) * scored[f"{row.factor}_z"]
    scored["rank"] = scored.groupby("date")["score"].rank(method="first", ascending=False)
    return scored


def _clip_and_normalize_weights(weights: dict[str, float], max_single_weight: float) -> dict[str, float]:
    if not weights:
        return {}
    clipped = {symbol: max(0.0, min(float(value), max_single_weight)) for symbol, value in weights.items()}
    total = sum(clipped.values())
    if total <= 1e-12:
        equal = 1.0 / len(clipped)
        return {symbol: equal for symbol in clipped}
    normalized = {symbol: value / total for symbol, value in clipped.items() if value > 1e-8}
    return normalized


def _apply_turnover_limit(
    target_weights: dict[str, float],
    previous_weights: dict[str, float],
    max_turnover: float,
) -> tuple[dict[str, float], float, float]:
    all_symbols = set(target_weights) | set(previous_weights)
    turnover = sum(abs(target_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in all_symbols)
    if turnover <= max_turnover or turnover <= 1e-12:
        return target_weights, turnover, 1.0
    alpha = max_turnover / turnover
    blended = {
        symbol: previous_weights.get(symbol, 0.0) + alpha * (target_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
        for symbol in all_symbols
    }
    blended = {symbol: weight for symbol, weight in blended.items() if weight > 1e-6}
    total = sum(blended.values())
    blended = {symbol: weight / total for symbol, weight in blended.items()}
    return blended, turnover, alpha


def _select_portfolio(
    group: pd.DataFrame,
    config: QuantStrategyConfig,
    previous_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    previous_weights = previous_weights or {}
    sorted_group = group.sort_values(["score", "market_cap_est", "symbol"], ascending=[False, False, True]).copy()
    eligible = sorted_group.loc[(sorted_group["limit_up_flag"] < 1.0) & (sorted_group["market_cap_rank"] >= config.min_market_cap_quantile)].copy()
    if eligible.empty:
        eligible = sorted_group.copy()

    picks: list[pd.Series] = []
    industry_counts: dict[str, int] = {}
    for _, row in eligible.iterrows():
        industry = str(row.get("industry", "未知行业"))
        if industry_counts.get(industry, 0) >= config.max_positions_per_industry:
            continue
        picks.append(row)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(picks) >= config.top_k:
            break

    if not picks:
        picks = [row for _, row in sorted_group.head(config.top_k).iterrows()]

    portfolio = pd.DataFrame(picks).copy().drop_duplicates(subset=["symbol"])
    score_floor = float(min(portfolio["score"].min(), 0.0))
    raw_weights = {
        row.symbol: float(row.score - score_floor + 1e-6)
        for row in portfolio.itertuples(index=False)
    }
    target_weights = _clip_and_normalize_weights(raw_weights, max_single_weight=config.max_single_weight)
    adjusted_weights, raw_turnover, alpha = _apply_turnover_limit(target_weights, previous_weights, config.max_turnover)

    adjusted = group.loc[group["symbol"].isin(adjusted_weights)].copy()
    adjusted["target_weight"] = adjusted["symbol"].map(adjusted_weights)
    adjusted = adjusted.sort_values("target_weight", ascending=False).reset_index(drop=True)
    adjusted.attrs["raw_turnover"] = raw_turnover
    adjusted.attrs["turnover_alpha"] = alpha
    return adjusted


def _build_backtest(
    scored_frame: pd.DataFrame,
    test_dates: list[pd.Timestamp],
    config: QuantStrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    test_frame = scored_frame.loc[scored_frame["date"].isin(test_dates)].copy()
    if test_frame.empty:
        raise ValueError("测试区间为空，无法回测策略。")

    trading_dates = sorted(pd.to_datetime(test_frame["date"]).drop_duplicates())
    daily_returns = test_frame.pivot_table(index="date", columns="symbol", values="stock_return_1")
    benchmark_daily = daily_returns.mean(axis=1).fillna(0.0)
    signal_dates = trading_dates[:: config.rebalance_frequency_days]

    daily_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}

    for idx, signal_date in enumerate(signal_dates):
        current = test_frame.loc[test_frame["date"] == signal_date].copy()
        portfolio = _select_portfolio(current, config=config, previous_weights=previous_weights)
        if portfolio.empty:
            continue

        new_weights = {row.symbol: float(row.target_weight) for row in portfolio.itertuples(index=False)}
        next_signal_date = signal_dates[idx + 1] if idx + 1 < len(signal_dates) else None
        holding_dates = [trade_date for trade_date in trading_dates if trade_date > signal_date and (next_signal_date is None or trade_date <= next_signal_date)]

        raw_turnover = float(portfolio.attrs.get("raw_turnover", 0.0))
        realized_turnover = sum(abs(new_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in set(previous_weights) | set(new_weights))
        sell_turnover = sum(max(previous_weights.get(symbol, 0.0) - new_weights.get(symbol, 0.0), 0.0) for symbol in set(previous_weights) | set(new_weights))
        trading_cost = realized_turnover * (config.transaction_cost_bps / 10000.0) + sell_turnover * (config.sell_tax_bps / 10000.0)

        period_strategy = 1.0
        period_benchmark = 1.0
        for offset, holding_date in enumerate(holding_dates):
            row = daily_returns.loc[holding_date].fillna(0.0)
            strategy_return = float(sum(new_weights.get(symbol, 0.0) * float(row.get(symbol, 0.0)) for symbol in row.index))
            if offset == 0:
                strategy_return -= trading_cost
            benchmark_return = float(benchmark_daily.loc[holding_date])
            period_strategy *= 1.0 + strategy_return
            period_benchmark *= 1.0 + benchmark_return
            daily_rows.append(
                {
                    "date": holding_date,
                    "signal_date": signal_date,
                    "strategy_return": strategy_return,
                    "benchmark_return": benchmark_return,
                    "active_symbols": ", ".join(f"{symbol}:{weight:.2%}" for symbol, weight in sorted(new_weights.items())),
                }
            )

        rebalance_rows.append(
            {
                "signal_date": signal_date,
                "hold_start_date": holding_dates[0] if holding_dates else pd.NaT,
                "hold_end_date": holding_dates[-1] if holding_dates else pd.NaT,
                "turnover": realized_turnover,
                "raw_turnover": raw_turnover,
                "sell_turnover": sell_turnover,
                "trading_cost": trading_cost,
                "holding_count": len(portfolio),
                "holdings": ", ".join(
                    f"{row.symbol} {row.name} {float(row.target_weight):.1%}" for row in portfolio.itertuples(index=False)
                ),
                "top_score": float(portfolio["score"].iloc[0]),
                "mean_score": float(portfolio["score"].mean()),
                "period_strategy_return": float(period_strategy - 1.0),
                "period_benchmark_return": float(period_benchmark - 1.0),
                "period_excess_return": float(period_strategy / period_benchmark - 1.0),
                "period_success": int(period_strategy > period_benchmark),
            }
        )
        previous_weights = new_weights

    daily_frame = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    if daily_frame.empty:
        raise ValueError("测试区间没有生成有效交易日收益。")
    daily_frame["strategy_curve"] = (1.0 + daily_frame["strategy_return"]).cumprod()
    daily_frame["benchmark_curve"] = (1.0 + daily_frame["benchmark_return"]).cumprod()
    daily_frame["excess_curve"] = daily_frame["strategy_curve"] / daily_frame["benchmark_curve"]
    strategy_peak = daily_frame["strategy_curve"].cummax()
    daily_frame["drawdown"] = daily_frame["strategy_curve"] / strategy_peak - 1.0

    rebalance_log = pd.DataFrame(rebalance_rows)
    return daily_frame, rebalance_log, previous_weights


def _safe_annualized_return(curve: pd.Series, periods: int) -> float:
    if periods <= 0 or curve.empty:
        return 0.0
    total_curve = float(curve.iloc[-1])
    if total_curve <= 0:
        return -1.0
    return total_curve ** (252.0 / periods) - 1.0


def _compute_metrics(backtest_daily: pd.DataFrame, rebalance_log: pd.DataFrame, fold_count: int) -> dict[str, Any]:
    strategy_returns = backtest_daily["strategy_return"]
    benchmark_returns = backtest_daily["benchmark_return"]
    excess_returns = strategy_returns - benchmark_returns
    strategy_curve = backtest_daily["strategy_curve"]
    benchmark_curve = backtest_daily["benchmark_curve"]

    strategy_vol = float(strategy_returns.std(ddof=0) * np.sqrt(252)) if len(strategy_returns) > 1 else 0.0
    benchmark_vol = float(benchmark_returns.std(ddof=0) * np.sqrt(252)) if len(benchmark_returns) > 1 else 0.0
    excess_vol = float(excess_returns.std(ddof=0) * np.sqrt(252)) if len(excess_returns) > 1 else 0.0

    annualized_return = _safe_annualized_return(strategy_curve, len(strategy_curve))
    benchmark_annualized_return = _safe_annualized_return(benchmark_curve, len(benchmark_curve))
    sharpe = annualized_return / strategy_vol if strategy_vol > 1e-12 else 0.0
    information_ratio = float(excess_returns.mean() * np.sqrt(252) / excess_returns.std(ddof=0)) if excess_vol > 1e-12 else 0.0
    signal_success_rate = float(rebalance_log["period_success"].mean()) if not rebalance_log.empty else 0.0

    return {
        "total_return": float(strategy_curve.iloc[-1] - 1.0),
        "benchmark_total_return": float(benchmark_curve.iloc[-1] - 1.0),
        "annualized_return": float(annualized_return),
        "benchmark_annualized_return": float(benchmark_annualized_return),
        "annualized_volatility": strategy_vol,
        "benchmark_annualized_volatility": benchmark_vol,
        "sharpe": float(sharpe),
        "information_ratio": float(information_ratio),
        "max_drawdown": float(backtest_daily["drawdown"].min()),
        "excess_return": float(strategy_curve.iloc[-1] / benchmark_curve.iloc[-1] - 1.0),
        "win_rate_vs_benchmark": float((strategy_returns > benchmark_returns).mean()),
        "average_turnover": float(rebalance_log["turnover"].mean()) if not rebalance_log.empty else 0.0,
        "signal_success_rate": signal_success_rate,
        "rebalance_count": int(len(rebalance_log)),
        "walk_forward_fold_count": int(fold_count),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_quant_strategy_artifacts(
    strategy_id: str,
    feature_frame: pd.DataFrame,
    scored_frame: pd.DataFrame,
    factor_ic_history: pd.DataFrame,
    factor_weights: pd.DataFrame,
    backtest_daily: pd.DataFrame,
    rebalance_log: pd.DataFrame,
    latest_portfolio: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    simulation_account: dict[str, Any],
    trade_plan: list[dict[str, Any]],
    ai_trade_context: dict[str, Any],
    metrics: dict[str, Any],
    run_summary: dict[str, Any],
) -> dict[str, str]:
    target_dir = artifact_dir(strategy_id)
    paths = {
        "feature_frame": target_dir / "feature_frame.csv",
        "scored_frame": target_dir / "scored_frame.csv",
        "factor_ic_history": target_dir / "factor_ic_history.csv",
        "factor_weights": target_dir / "factor_weights.csv",
        "backtest_daily": target_dir / "backtest_daily.csv",
        "rebalance_log": target_dir / "rebalance_log.csv",
        "latest_portfolio": target_dir / "latest_portfolio.csv",
        "fold_metrics": target_dir / "walk_forward_folds.csv",
        "metrics": target_dir / "metrics.json",
        "run_summary": target_dir / "run_summary.json",
        "simulation_account": target_dir / "simulation_account.json",
        "trade_plan": target_dir / "trade_plan.json",
        "ai_trade_context": target_dir / "ai_trade_context.json",
    }
    feature_frame.to_csv(paths["feature_frame"], index=False, encoding="utf-8-sig")
    scored_frame.to_csv(paths["scored_frame"], index=False, encoding="utf-8-sig")
    factor_ic_history.to_csv(paths["factor_ic_history"], index=False, encoding="utf-8-sig")
    factor_weights.to_csv(paths["factor_weights"], index=False, encoding="utf-8-sig")
    backtest_daily.to_csv(paths["backtest_daily"], index=False, encoding="utf-8-sig")
    rebalance_log.to_csv(paths["rebalance_log"], index=False, encoding="utf-8-sig")
    latest_portfolio.to_csv(paths["latest_portfolio"], index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(paths["fold_metrics"], index=False, encoding="utf-8-sig")
    _write_json(paths["metrics"], metrics)
    _write_json(paths["run_summary"], run_summary)
    _write_json(paths["simulation_account"], simulation_account)
    paths["trade_plan"].write_text(json.dumps(trade_plan, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_json(paths["ai_trade_context"], ai_trade_context)
    return {name: str(path) for name, path in paths.items()}


def load_quant_strategy_artifacts(strategy_id: str = DEFAULT_STRATEGY_ID) -> dict[str, Any]:
    target_dir = artifact_dir(strategy_id)
    return {
        "feature_frame": pd.read_csv(
            target_dir / "feature_frame.csv",
            parse_dates=["date", "report_date", "fundamental_asof_date", "dividend_announce_date"],
            dtype={"symbol": str},
        ),
        "scored_frame": pd.read_csv(
            target_dir / "scored_frame.csv",
            parse_dates=["date", "report_date", "fundamental_asof_date", "dividend_announce_date"],
            dtype={"symbol": str},
        ),
        "factor_ic_history": pd.read_csv(target_dir / "factor_ic_history.csv", parse_dates=["date"]),
        "factor_weights": pd.read_csv(target_dir / "factor_weights.csv"),
        "backtest_daily": pd.read_csv(target_dir / "backtest_daily.csv", parse_dates=["date", "signal_date"]),
        "rebalance_log": pd.read_csv(target_dir / "rebalance_log.csv", parse_dates=["signal_date", "hold_start_date", "hold_end_date"]),
        "latest_portfolio": pd.read_csv(target_dir / "latest_portfolio.csv", parse_dates=["date"], dtype={"symbol": str}),
        "fold_metrics": pd.read_csv(target_dir / "walk_forward_folds.csv", parse_dates=["train_end_date", "test_start_date", "test_end_date"]),
        "metrics": _read_json(target_dir / "metrics.json"),
        "run_summary": _read_json(target_dir / "run_summary.json"),
        "simulation_account": _read_json(target_dir / "simulation_account.json"),
        "trade_plan": _read_json(target_dir / "trade_plan.json") if (target_dir / "trade_plan.json").exists() else [],
        "ai_trade_context": _read_json(target_dir / "ai_trade_context.json") if (target_dir / "ai_trade_context.json").exists() else {},
        "artifact_dir": str(target_dir),
    }


def _walk_forward_splits(unique_dates: list[pd.Timestamp], config: QuantStrategyConfig) -> list[tuple[pd.Timestamp, list[pd.Timestamp]]]:
    splits: list[tuple[pd.Timestamp, list[pd.Timestamp]]] = []
    if len(unique_dates) <= config.holding_period_days + config.initial_train_days:
        return splits

    start_idx = max(config.initial_train_days, int(len(unique_dates) * config.train_ratio))
    last_idx = len(unique_dates) - config.holding_period_days - 1
    while start_idx < last_idx:
        train_end_date = pd.Timestamp(unique_dates[start_idx - 1])
        test_end_idx = min(start_idx + config.walk_forward_test_days - 1, last_idx)
        test_dates = [pd.Timestamp(value) for value in unique_dates[start_idx : test_end_idx + 1]]
        if len(test_dates) >= config.rebalance_frequency_days:
            splits.append((train_end_date, test_dates))
        start_idx += config.walk_forward_step_days
    return splits


def run_quant_strategy_pipeline(
    symbol: str = DEFAULT_SYMBOL,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    force_refresh: bool = False,
    config: QuantStrategyConfig | None = None,
) -> dict[str, Any]:
    config = config or QuantStrategyConfig()
    _set_seed(config.seed)

    feature_frame = _load_or_build_feature_frame(symbol=symbol, start_date=start_date, end_date=end_date, force_refresh=force_refresh, config=config)
    factor_frame, factor_columns = _prepare_factor_frame(feature_frame, config)

    unique_dates = sorted(pd.to_datetime(factor_frame["date"]).drop_duplicates())
    splits = _walk_forward_splits(unique_dates, config)
    if not splits:
        raise ValueError("有效交易日不足，无法构建滚动 walk-forward 验证。")

    scored_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    rebalance_frames: list[pd.DataFrame] = []
    ic_frames: list[pd.DataFrame] = []
    fold_weight_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for fold_id, (train_end_date, test_dates) in enumerate(splits, start=1):
        train_frame = factor_frame.loc[(factor_frame["date"] <= train_end_date) & factor_frame["future_excess_return"].notna()].copy()
        test_frame = factor_frame.loc[factor_frame["date"].isin(test_dates)].copy()
        factor_ic_history = _compute_factor_ic_history(train_frame, factor_columns, config.min_cross_section)
        factor_weights = _derive_factor_weights(factor_ic_history)
        scored_test = _apply_factor_scores(test_frame, factor_weights)
        backtest_daily, rebalance_log, _ = _build_backtest(scored_test, test_dates=test_dates, config=config)
        fold_metrics = _compute_metrics(backtest_daily, rebalance_log, fold_count=1)

        scored_test["fold_id"] = fold_id
        factor_ic_history["fold_id"] = fold_id
        factor_weights["fold_id"] = fold_id
        backtest_daily["fold_id"] = fold_id
        rebalance_log["fold_id"] = fold_id

        scored_frames.append(scored_test)
        ic_frames.append(factor_ic_history)
        fold_weight_frames.append(factor_weights)
        daily_frames.append(backtest_daily)
        rebalance_frames.append(rebalance_log)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_end_date": train_end_date,
                "test_start_date": test_dates[0],
                "test_end_date": test_dates[-1],
                **fold_metrics,
            }
        )

    scored_frame = pd.concat(scored_frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    factor_ic_history = pd.concat(ic_frames, ignore_index=True).sort_values(["date", "factor", "fold_id"]).reset_index(drop=True)
    factor_weights = (
        pd.concat(fold_weight_frames, ignore_index=True)
        .sort_values(["fold_id", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )
    backtest_daily = pd.concat(daily_frames, ignore_index=True).sort_values(["date", "fold_id"]).reset_index(drop=True)
    rebalance_log = pd.concat(rebalance_frames, ignore_index=True).sort_values(["signal_date", "fold_id"]).reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_rows).sort_values("fold_id").reset_index(drop=True)
    metrics = _compute_metrics(backtest_daily, rebalance_log, fold_count=len(fold_metrics))

    latest_train_end = pd.Timestamp(unique_dates[-config.holding_period_days - 1])
    latest_train_frame = factor_frame.loc[(factor_frame["date"] <= latest_train_end) & factor_frame["future_excess_return"].notna()].copy()
    latest_weights = _derive_factor_weights(_compute_factor_ic_history(latest_train_frame, factor_columns, config.min_cross_section))
    current_features = factor_frame.loc[factor_frame["date"] == factor_frame["date"].max()].copy()
    latest_scored = _apply_factor_scores(current_features, latest_weights)
    latest_portfolio = _select_portfolio(latest_scored, config=config, previous_weights={})
    latest_portfolio = latest_portfolio[["date", "symbol", "name", "industry", "score", "rank", "target_weight"]].reset_index(drop=True)

    latest_prices = feature_frame.loc[feature_frame["date"] == feature_frame["date"].max(), ["symbol", "close"]].copy()
    account_path = artifact_dir(config.strategy_id) / "simulation_account.json"
    simulation_account, trade_plan = rebalance_paper_account(
        account_path=account_path,
        latest_portfolio=latest_portfolio,
        price_frame=latest_prices,
        initial_cash=config.initial_cash,
        lot_size=config.lot_size,
    )

    latest_trade_date = pd.Timestamp(feature_frame["date"].max()).date().isoformat()
    latest_available_symbols = (
        feature_frame.sort_values("date")
        .groupby("symbol")[["name", "industry"]]
        .last()
        .reset_index()
        .sort_values("symbol")
    )

    ai_trade_context = build_ai_trade_context(
        summary={
            "strategy_id": config.strategy_id,
            "symbol": symbol,
            "latest_trade_date": latest_trade_date,
            "latest_holdings": latest_portfolio.to_dict(orient="records"),
        },
        metrics=metrics,
        trade_plan=trade_plan,
        config=config.to_dict(),
    )

    run_summary = {
        "strategy_id": config.strategy_id,
        "symbol": symbol,
        "display_name": DISPLAY_NAME,
        "latest_trade_date": latest_trade_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_end_date": latest_train_end.date().isoformat(),
        "holding_period_days": config.holding_period_days,
        "rebalance_frequency_days": config.rebalance_frequency_days,
        "top_k": config.top_k,
        "latest_holdings": latest_portfolio.to_dict(orient="records"),
        "stock_pool": latest_available_symbols.to_dict(orient="records"),
        "status_message": (
            f"基于更大范围 A 股样本池做多因子横截面排序，滚动 walk-forward 验证，"
            f"每 {config.rebalance_frequency_days} 个交易日调仓一次，当前为 {config.broker_mode} 模式模拟盘。"
        ),
        "factor_columns": factor_columns,
        "config": config.to_dict(),
        "broker_mode": config.broker_mode,
        "live_trading_enabled": False,
        "trade_review_required": True,
        "success_rate": metrics["signal_success_rate"],
    }

    artifact_paths = save_quant_strategy_artifacts(
        strategy_id=config.strategy_id,
        feature_frame=feature_frame,
        scored_frame=scored_frame,
        factor_ic_history=factor_ic_history,
        factor_weights=factor_weights,
        backtest_daily=backtest_daily,
        rebalance_log=rebalance_log,
        latest_portfolio=latest_portfolio,
        fold_metrics=fold_metrics,
        simulation_account=simulation_account,
        trade_plan=trade_plan,
        ai_trade_context=ai_trade_context,
        metrics=metrics,
        run_summary=run_summary,
    )
    dashboard = load_quant_strategy_artifacts(config.strategy_id)
    site_data_path = export_static_site(dashboard)
    dashboard["artifact_paths"] = artifact_paths
    dashboard["site_data_path"] = site_data_path
    return dashboard
