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
    POWER_STOCK_POOL,
    QuantStrategyConfig,
    TrainingConfig,
    artifact_dir,
    feature_cache_path,
)
from .data import fetch_market_data
from .features import build_features
from .site_export import export_static_site

FACTOR_SPECS: list[tuple[str, str]] = [
    ("relative_strength_5", "5日相对强度"),
    ("trend_strength", "20日趋势强度"),
    ("macd_trend", "MACD趋势"),
    ("low_volatility", "低波动"),
    ("rsi_balance", "RSI均衡"),
    ("volume_support", "量能支持"),
    ("turnover_stability", "换手稳定"),
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_or_build_feature_frame(
    symbol: str,
    start_date: str,
    end_date: str | None,
    force_refresh: bool,
    holding_period_days: int,
) -> pd.DataFrame:
    cache_symbol = f"{symbol}_pool_quant_h{holding_period_days}"
    cache_path = feature_cache_path(cache_symbol, start_date, end_date)
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, parse_dates=["date"], dtype={"symbol": str})

    market_data = fetch_market_data(symbol=symbol, start=start_date, end=end_date, force_refresh=force_refresh)
    build_config = TrainingConfig(prediction_horizon_days=holding_period_days, return_threshold=0.0, verbose=False)
    frames: list[pd.DataFrame] = []
    for stock_symbol, stock_frame in market_data.stocks.items():
        frame, _ = build_features(
            stock_frame,
            market_data.indices,
            market_data.industry_proxy,
            market_data.northbound,
            config=build_config,
            symbol=stock_symbol,
        )
        if len(frame) >= 80:
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    combined.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return combined


def _prepare_factor_frame(feature_frame: pd.DataFrame, config: QuantStrategyConfig) -> tuple[pd.DataFrame, list[str]]:
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)

    market_proxy = frame[["date", "power_index_close"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    market_proxy["future_power_return"] = market_proxy["power_index_close"].shift(-config.holding_period_days) / market_proxy["power_index_close"] - 1
    frame = frame.merge(market_proxy[["date", "future_power_return"]], on="date", how="left")
    frame["future_excess_return"] = frame["future_return_horizon"] - frame["future_power_return"]

    frame["relative_strength_5"] = frame["stock_return_5"] - frame["power_return_5"]
    frame["trend_strength"] = frame["ma_20_dev"]
    frame["macd_trend"] = frame["macd_hist"]
    frame["low_volatility"] = -frame["volatility_20"]
    frame["rsi_balance"] = -(frame["rsi_14"] - 55.0).abs() / 100.0
    frame["volume_support"] = frame["volume_ratio_5"]
    frame["turnover_stability"] = -frame["stock_turnover_change_1"].abs()

    factor_columns = [factor_name for factor_name, _ in FACTOR_SPECS]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=factor_columns + ["future_excess_return", "stock_return_1"]).reset_index(drop=True)

    for factor_name in factor_columns:
        frame[f"{factor_name}_z"] = frame.groupby("date")[factor_name].transform(
            lambda values: (values - values.mean()) / values.std(ddof=0) if float(values.std(ddof=0)) > 1e-12 else 0.0
        )

    return frame, factor_columns


def _compute_factor_ic_history(frame: pd.DataFrame, factor_columns: list[str], min_cross_section: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("date", sort=True):
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
    label_map = dict(FACTOR_SPECS)
    summary["factor_label"] = summary["factor"].map(label_map)
    return summary.sort_values("weight", ascending=False).reset_index(drop=True)


def _apply_factor_scores(frame: pd.DataFrame, factor_weights: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = 0.0
    for row in factor_weights.itertuples(index=False):
        scored["score"] += float(row.weight) * scored[f"{row.factor}_z"]
    scored["rank"] = scored.groupby("date")["score"].rank(method="first", ascending=False)
    return scored


def _select_portfolio(group: pd.DataFrame, top_k: int) -> pd.DataFrame:
    eligible = group.loc[group["limit_up_flag"] < 1.0].copy()
    if len(eligible) < top_k:
        eligible = group.copy()
    return eligible.sort_values(["score", "symbol"], ascending=[False, True]).head(top_k).copy()


def _build_backtest(
    scored_frame: pd.DataFrame,
    test_start_date: pd.Timestamp,
    config: QuantStrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_frame = scored_frame.loc[scored_frame["date"] > test_start_date].copy()
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
        portfolio = _select_portfolio(current, config.top_k)
        if portfolio.empty:
            continue

        new_weights = {row.symbol: 1.0 / len(portfolio) for row in portfolio.itertuples(index=False)}
        next_signal_date = signal_dates[idx + 1] if idx + 1 < len(signal_dates) else None
        holding_dates = [trade_date for trade_date in trading_dates if trade_date > signal_date and (next_signal_date is None or trade_date <= next_signal_date)]

        turnover = sum(abs(new_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in set(previous_weights) | set(new_weights))
        sell_turnover = sum(max(previous_weights.get(symbol, 0.0) - new_weights.get(symbol, 0.0), 0.0) for symbol in set(previous_weights) | set(new_weights))
        trading_cost = turnover * (config.transaction_cost_bps / 10000.0) + sell_turnover * (config.sell_tax_bps / 10000.0)

        rebalance_rows.append(
            {
                "signal_date": signal_date,
                "hold_start_date": holding_dates[0] if holding_dates else pd.NaT,
                "hold_end_date": holding_dates[-1] if holding_dates else pd.NaT,
                "turnover": turnover,
                "sell_turnover": sell_turnover,
                "trading_cost": trading_cost,
                "holding_count": len(portfolio),
                "holdings": ", ".join(f"{row.symbol} {row.name}" for row in portfolio.itertuples(index=False)),
                "top_score": float(portfolio["score"].iloc[0]),
                "mean_score": float(portfolio["score"].mean()),
            }
        )

        for offset, holding_date in enumerate(holding_dates):
            row = daily_returns.loc[holding_date].fillna(0.0)
            strategy_return = float(sum(new_weights.get(symbol, 0.0) * float(row.get(symbol, 0.0)) for symbol in row.index))
            if offset == 0:
                strategy_return -= trading_cost
            benchmark_return = float(benchmark_daily.loc[holding_date])
            daily_rows.append(
                {
                    "date": holding_date,
                    "strategy_return": strategy_return,
                    "benchmark_return": benchmark_return,
                    "active_symbols": ", ".join(sorted(new_weights)),
                }
            )
        previous_weights = new_weights

    daily_frame = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    daily_frame["strategy_curve"] = (1.0 + daily_frame["strategy_return"]).cumprod()
    daily_frame["benchmark_curve"] = (1.0 + daily_frame["benchmark_return"]).cumprod()
    daily_frame["excess_curve"] = daily_frame["strategy_curve"] / daily_frame["benchmark_curve"]
    strategy_peak = daily_frame["strategy_curve"].cummax()
    daily_frame["drawdown"] = daily_frame["strategy_curve"] / strategy_peak - 1.0

    rebalance_log = pd.DataFrame(rebalance_rows)
    latest_date = scored_frame["date"].max()
    latest_portfolio = _select_portfolio(scored_frame.loc[scored_frame["date"] == latest_date].copy(), config.top_k)
    latest_portfolio = latest_portfolio[["date", "symbol", "name", "score", "rank"]].reset_index(drop=True)
    return daily_frame, rebalance_log, latest_portfolio


def _safe_annualized_return(curve: pd.Series, periods: int) -> float:
    if periods <= 0 or curve.empty:
        return 0.0
    total_curve = float(curve.iloc[-1])
    if total_curve <= 0:
        return -1.0
    return total_curve ** (252.0 / periods) - 1.0


def _compute_metrics(backtest_daily: pd.DataFrame, rebalance_log: pd.DataFrame) -> dict[str, Any]:
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
        "rebalance_count": int(len(rebalance_log)),
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
        "metrics": target_dir / "metrics.json",
        "run_summary": target_dir / "run_summary.json",
    }
    feature_frame.to_csv(paths["feature_frame"], index=False, encoding="utf-8-sig")
    scored_frame.to_csv(paths["scored_frame"], index=False, encoding="utf-8-sig")
    factor_ic_history.to_csv(paths["factor_ic_history"], index=False, encoding="utf-8-sig")
    factor_weights.to_csv(paths["factor_weights"], index=False, encoding="utf-8-sig")
    backtest_daily.to_csv(paths["backtest_daily"], index=False, encoding="utf-8-sig")
    rebalance_log.to_csv(paths["rebalance_log"], index=False, encoding="utf-8-sig")
    latest_portfolio.to_csv(paths["latest_portfolio"], index=False, encoding="utf-8-sig")
    _write_json(paths["metrics"], metrics)
    _write_json(paths["run_summary"], run_summary)
    return {name: str(path) for name, path in paths.items()}


def load_quant_strategy_artifacts(strategy_id: str = DEFAULT_STRATEGY_ID) -> dict[str, Any]:
    target_dir = artifact_dir(strategy_id)
    return {
        "feature_frame": pd.read_csv(target_dir / "feature_frame.csv", parse_dates=["date"], dtype={"symbol": str}),
        "scored_frame": pd.read_csv(target_dir / "scored_frame.csv", parse_dates=["date"], dtype={"symbol": str}),
        "factor_ic_history": pd.read_csv(target_dir / "factor_ic_history.csv", parse_dates=["date"]),
        "factor_weights": pd.read_csv(target_dir / "factor_weights.csv"),
        "backtest_daily": pd.read_csv(target_dir / "backtest_daily.csv", parse_dates=["date"]),
        "rebalance_log": pd.read_csv(target_dir / "rebalance_log.csv", parse_dates=["signal_date", "hold_start_date", "hold_end_date"]),
        "latest_portfolio": pd.read_csv(target_dir / "latest_portfolio.csv", parse_dates=["date"], dtype={"symbol": str}),
        "metrics": _read_json(target_dir / "metrics.json"),
        "run_summary": _read_json(target_dir / "run_summary.json"),
        "artifact_dir": str(target_dir),
    }


def run_quant_strategy_pipeline(
    symbol: str = DEFAULT_SYMBOL,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    force_refresh: bool = False,
    config: QuantStrategyConfig | None = None,
) -> dict[str, Any]:
    config = config or QuantStrategyConfig()
    _set_seed(config.seed)

    feature_frame = _load_or_build_feature_frame(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
        holding_period_days=config.holding_period_days,
    )
    factor_frame, factor_columns = _prepare_factor_frame(feature_frame, config)

    unique_dates = sorted(pd.to_datetime(factor_frame["date"]).drop_duplicates())
    train_count = max(config.min_cross_section, int(len(unique_dates) * config.train_ratio))
    train_count = min(train_count, len(unique_dates) - config.rebalance_frequency_days - 1)
    if train_count < config.min_cross_section:
        raise ValueError("有效交易日不足，无法构建训练/测试切分。")

    train_end_date = pd.Timestamp(unique_dates[train_count - 1])
    train_frame = factor_frame.loc[factor_frame["date"] <= train_end_date].copy()

    factor_ic_history = _compute_factor_ic_history(train_frame, factor_columns, config.min_cross_section)
    factor_weights = _derive_factor_weights(factor_ic_history)
    scored_frame = _apply_factor_scores(factor_frame, factor_weights)

    name_map = POWER_STOCK_POOL.copy()
    scored_frame["name"] = scored_frame["symbol"].map(name_map).fillna(scored_frame["symbol"])
    backtest_daily, rebalance_log, latest_portfolio = _build_backtest(scored_frame, train_end_date, config)
    latest_portfolio["name"] = latest_portfolio["symbol"].map(name_map).fillna(latest_portfolio["symbol"])

    metrics = _compute_metrics(backtest_daily, rebalance_log)
    latest_trade_date = pd.Timestamp(scored_frame["date"].max()).date().isoformat()

    run_summary = {
        "strategy_id": config.strategy_id,
        "symbol": symbol,
        "display_name": DISPLAY_NAME,
        "latest_trade_date": latest_trade_date,
        "train_end_date": train_end_date.date().isoformat(),
        "holding_period_days": config.holding_period_days,
        "rebalance_frequency_days": config.rebalance_frequency_days,
        "top_k": config.top_k,
        "latest_holdings": latest_portfolio.to_dict(orient="records"),
        "stock_pool": [{"symbol": key, "name": value} for key, value in POWER_STOCK_POOL.items()],
        "status_message": f"基于电力股样本池做多因子横截面排序，每 {config.rebalance_frequency_days} 个交易日调仓一次，等权持有前 {config.top_k} 名。",
        "factor_columns": factor_columns,
        "config": config.to_dict(),
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
        metrics=metrics,
        run_summary=run_summary,
    )
    dashboard = load_quant_strategy_artifacts(config.strategy_id)
    site_data_path = export_static_site(dashboard)
    dashboard["artifact_paths"] = artifact_paths
    dashboard["site_data_path"] = site_data_path
    return dashboard
