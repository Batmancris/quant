from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_SYMBOL, INDEX_PREFIXES, TrainingConfig

FEATURE_COLUMNS = [
    "stock_return_1",
    "stock_return_3",
    "stock_return_5",
    "stock_log_return_1",
    "stock_amplitude_pct",
    "stock_body_pct",
    "stock_volume_change_1",
    "stock_turnover_change_1",
    "volume_ratio_5",
    "ma_5_dev",
    "ma_10_dev",
    "ma_20_dev",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bollinger_position",
    "limit_up_flag",
    "limit_down_flag",
    "sse_return_1",
    "sse_return_5",
    "sse_volatility_10",
    "szse_return_1",
    "szse_return_5",
    "szse_volatility_10",
    "hs300_return_1",
    "hs300_return_5",
    "hs300_volatility_10",
    "power_return_1",
    "power_return_5",
    "power_volatility_10",
    "power_up_ratio",
    "power_limit_up_ratio",
    "power_volume_change_1",
    "power_turnover_change_1",
    "north_net_deal",
    "north_net_deal_5",
    "north_inflow",
    "north_inflow_5",
    "north_cumulative",
    "north_cumulative_change_5",
    "market_same_direction",
]


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def _compute_bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    band_width = (upper - lower).replace(0, np.nan)
    return ((close - lower) / band_width).replace([np.inf, -np.inf], np.nan)


def _prepare_index_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    prepared = frame[["date", "close"]].copy()
    prepared[f"{prefix}_return_1"] = prepared["close"].pct_change()
    prepared[f"{prefix}_return_5"] = prepared["close"].pct_change(5)
    prepared[f"{prefix}_volatility_10"] = prepared[f"{prefix}_return_1"].rolling(10, min_periods=10).std()
    return prepared[["date", f"{prefix}_return_1", f"{prefix}_return_5", f"{prefix}_volatility_10"]]


def _prepare_northbound_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "north_net_deal",
                "north_net_deal_5",
                "north_inflow",
                "north_inflow_5",
                "north_cumulative",
                "north_cumulative_change_5",
            ]
        )
    prepared = frame[["date", "north_net_deal", "north_inflow", "north_cumulative"]].copy().sort_values("date")
    for column in ["north_net_deal", "north_inflow", "north_cumulative"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").ffill().fillna(0.0)
    prepared["north_net_deal_5"] = prepared["north_net_deal"].rolling(5, min_periods=1).mean()
    prepared["north_inflow_5"] = prepared["north_inflow"].rolling(5, min_periods=1).mean()
    prepared["north_cumulative_change_5"] = prepared["north_cumulative"].diff(5)
    return prepared


def build_features(
    df_stock: pd.DataFrame,
    df_indices: dict[str, pd.DataFrame],
    industry_proxy: pd.DataFrame,
    northbound_frame: pd.DataFrame,
    config: TrainingConfig,
    symbol: str = DEFAULT_SYMBOL,
) -> tuple[pd.DataFrame, list[str]]:
    stock = df_stock.copy()
    stock["date"] = pd.to_datetime(stock["date"])
    stock = stock.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    stock["symbol"] = symbol
    stock["stock_return_1"] = stock["close"].pct_change()
    stock["stock_return_3"] = stock["close"].pct_change(3)
    stock["stock_return_5"] = stock["close"].pct_change(5)
    stock["stock_log_return_1"] = np.log(stock["close"]).diff()
    stock["stock_amplitude_pct"] = (stock["high"] - stock["low"]) / stock["close"].replace(0, np.nan)
    stock["stock_body_pct"] = (stock["close"] - stock["open"]) / stock["open"].replace(0, np.nan)
    stock["stock_volume_change_1"] = stock["volume"].pct_change()
    stock["stock_turnover_change_1"] = stock["turnover_rate"].pct_change()
    stock["volume_ratio_5"] = stock["volume"] / stock["volume"].rolling(5, min_periods=5).mean().shift(1)
    stock["ma_5_dev"] = stock["close"] / stock["close"].rolling(5, min_periods=5).mean() - 1
    stock["ma_10_dev"] = stock["close"] / stock["close"].rolling(10, min_periods=10).mean() - 1
    stock["ma_20_dev"] = stock["close"] / stock["close"].rolling(20, min_periods=20).mean() - 1
    stock["volatility_5"] = stock["stock_return_1"].rolling(5, min_periods=5).std()
    stock["volatility_10"] = stock["stock_return_1"].rolling(10, min_periods=10).std()
    stock["volatility_20"] = stock["stock_return_1"].rolling(20, min_periods=20).std()
    stock["rsi_14"] = _compute_rsi(stock["close"], window=14)
    stock["macd"], stock["macd_signal"], stock["macd_hist"] = _compute_macd(stock["close"])
    stock["bollinger_position"] = _compute_bollinger_position(stock["close"], window=20)
    stock["limit_up_flag"] = (stock["pct_change"] >= 9.5).astype(float)
    stock["limit_down_flag"] = (stock["pct_change"] <= -9.5).astype(float)

    merged = stock[
        [
            "date",
            "symbol",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "turnover_rate",
            "stock_return_1",
            "stock_return_3",
            "stock_return_5",
            "stock_log_return_1",
            "stock_amplitude_pct",
            "stock_body_pct",
            "stock_volume_change_1",
            "stock_turnover_change_1",
            "volume_ratio_5",
            "ma_5_dev",
            "ma_10_dev",
            "ma_20_dev",
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bollinger_position",
            "limit_up_flag",
            "limit_down_flag",
        ]
    ].copy()

    for index_symbol, index_frame in df_indices.items():
        prefix = INDEX_PREFIXES[index_symbol]
        merged = merged.merge(_prepare_index_features(index_frame.copy(), prefix), on="date", how="inner")

    merged = merged.merge(industry_proxy.copy(), on="date", how="left")
    merged = merged.merge(_prepare_northbound_features(northbound_frame.copy()), on="date", how="left")
    northbound_columns = [
        "north_net_deal",
        "north_net_deal_5",
        "north_inflow",
        "north_inflow_5",
        "north_cumulative",
        "north_cumulative_change_5",
    ]
    for column in northbound_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").ffill().fillna(0.0)

    merged["market_same_direction"] = (
        (np.sign(merged["stock_return_1"]) == np.sign(merged["sse_return_1"]))
        & (np.sign(merged["stock_return_1"]) == np.sign(merged["hs300_return_1"]))
    ).astype(float)
    merged["future_return_horizon"] = merged["close"].shift(-config.prediction_horizon_days) / merged["close"] - 1
    merged["label"] = np.where(
        merged["future_return_horizon"].notna(),
        (merged["future_return_horizon"] > config.return_threshold).astype(int),
        np.nan,
    )

    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged["date"])
    return merged, FEATURE_COLUMNS.copy()
