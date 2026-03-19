from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import akshare as ak
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from greenpower_demo.config import (
        DEFAULT_START_DATE,
        DEFAULT_SYMBOL,
        INDEX_DISPLAY_NAMES,
        POWER_STOCK_POOL,
        raw_cache_path,
        trade_calendar_cache_path,
    )
else:
    from .config import (
        DEFAULT_START_DATE,
        DEFAULT_SYMBOL,
        INDEX_DISPLAY_NAMES,
        POWER_STOCK_POOL,
        raw_cache_path,
        trade_calendar_cache_path,
    )


@dataclass(slots=True)
class MarketData:
    target_stock: pd.DataFrame
    stocks: dict[str, pd.DataFrame]
    indices: dict[str, pd.DataFrame]
    northbound: pd.DataFrame
    industry_proxy: pd.DataFrame
    trade_calendar: pd.DatetimeIndex


def _retry_fetch(fetcher: Callable[[], pd.DataFrame], description: str, retries: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fetcher()
        except Exception as exc:  # pragma: no cover - exercised during live runs
            last_error = exc
            if attempt == retries:
                break
            time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to fetch {description}") from last_error


def _stock_symbol_for_ak(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"


def _normalize_stock_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
            "turnover": "turnover_rate",
        }
    ).copy()
    renamed["date"] = pd.to_datetime(renamed["date"])
    renamed["symbol"] = symbol
    for column in ["open", "close", "high", "low", "volume", "amount", "turnover_rate"]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    renamed["amplitude_pct"] = (renamed["high"] - renamed["low"]) / renamed["close"].replace(0, pd.NA)
    renamed["pct_change"] = renamed["close"].pct_change() * 100
    renamed["change"] = renamed["close"].diff()
    return renamed.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _normalize_index_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    renamed = frame.copy()
    renamed["date"] = pd.to_datetime(renamed["date"])
    renamed["symbol"] = symbol
    if "amount" not in renamed.columns:
        renamed["amount"] = pd.NA
    for column in ["open", "close", "high", "low", "volume", "amount"]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _normalize_northbound_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "north_net_deal",
                "north_buy_amount",
                "north_sell_amount",
                "north_cumulative",
                "north_inflow",
                "north_balance",
                "north_hold_value",
                "hs300_close_ref",
                "hs300_pct_ref",
            ]
        )
    renamed = frame.rename(
        columns={
            "日期": "date",
            "当日成交净买额": "north_net_deal",
            "买入成交额": "north_buy_amount",
            "卖出成交额": "north_sell_amount",
            "历史累计净买额": "north_cumulative",
            "当日资金流入": "north_inflow",
            "当日余额": "north_balance",
            "持股市值": "north_hold_value",
            "沪深300": "hs300_close_ref",
            "沪深300-涨跌幅": "hs300_pct_ref",
        }
    ).copy()
    renamed["date"] = pd.to_datetime(renamed["date"])
    numeric_columns = [
        "north_net_deal",
        "north_buy_amount",
        "north_sell_amount",
        "north_cumulative",
        "north_inflow",
        "north_balance",
        "north_hold_value",
        "hs300_close_ref",
        "hs300_pct_ref",
    ]
    for column in numeric_columns:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _normalize_trade_calendar(frame: pd.DataFrame) -> pd.DatetimeIndex:
    trade_dates = pd.to_datetime(frame["trade_date"]).sort_values().drop_duplicates()
    return pd.DatetimeIndex(trade_dates)


def _load_or_fetch_csv(
    path: Path,
    fetcher: Callable[[], pd.DataFrame],
    description: str,
    force_refresh: bool,
    date_column: str,
    allow_empty_on_failure: bool = False,
) -> pd.DataFrame:
    if path.exists() and not force_refresh:
        return pd.read_csv(path, parse_dates=[date_column])
    try:
        frame = _retry_fetch(fetcher, description=description)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return frame
    except Exception as exc:
        if path.exists():
            warnings.warn(f"{description} refresh failed, using cached data: {exc}", stacklevel=2)
            return pd.read_csv(path, parse_dates=[date_column])
        if allow_empty_on_failure:
            warnings.warn(f"{description} unavailable, continuing without it: {exc}", stacklevel=2)
            return pd.DataFrame(columns=[date_column])
        raise


def fetch_stock_history(
    symbol: str = DEFAULT_SYMBOL,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_path = raw_cache_path("stock", symbol, start_date, end_date)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date or date.today().isoformat())

    def _fetch() -> pd.DataFrame:
        frame = ak.stock_zh_a_daily(symbol=_stock_symbol_for_ak(symbol), adjust="qfq")
        normalized = _normalize_stock_history(frame, symbol)
        mask = (normalized["date"] >= start_ts) & (normalized["date"] <= end_ts)
        return normalized.loc[mask].reset_index(drop=True)

    return _load_or_fetch_csv(cache_path, _fetch, f"stock history for {symbol}", force_refresh, "date")


def fetch_index_history(
    symbol: str,
    start_date: str,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_path = raw_cache_path("index", symbol, start_date, end_date)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date or date.today().isoformat())

    def _fetch() -> pd.DataFrame:
        frame = ak.stock_zh_index_daily(symbol=symbol)
        normalized = _normalize_index_history(frame, symbol)
        mask = (normalized["date"] >= start_ts) & (normalized["date"] <= end_ts)
        return normalized.loc[mask].reset_index(drop=True)

    return _load_or_fetch_csv(cache_path, _fetch, f"index history for {symbol}", force_refresh, "date")


def fetch_northbound_history(
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_path = raw_cache_path("northbound", "northbound", start_date, end_date)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date or date.today().isoformat())

    def _fetch() -> pd.DataFrame:
        frame = ak.stock_hsgt_hist_em()
        normalized = _normalize_northbound_history(frame)
        mask = (normalized["date"] >= start_ts) & (normalized["date"] <= end_ts)
        return normalized.loc[mask].reset_index(drop=True)

    frame = _load_or_fetch_csv(
        cache_path,
        _fetch,
        "northbound history",
        force_refresh,
        "date",
        allow_empty_on_failure=True,
    )
    if frame.empty:
        return _normalize_northbound_history(frame)
    return frame


def fetch_trade_calendar(force_refresh: bool = False) -> pd.DatetimeIndex:
    cache_path = trade_calendar_cache_path()

    def _fetch() -> pd.DataFrame:
        return ak.tool_trade_date_hist_sina()

    frame = _load_or_fetch_csv(cache_path, _fetch, "A-share trade calendar", force_refresh, "trade_date")
    return _normalize_trade_calendar(frame)


def build_power_industry_proxy(stocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, frame in stocks.items():
        part = frame[["date", "close", "volume", "turnover_rate", "pct_change"]].copy()
        part["symbol"] = symbol
        part["stock_return_1"] = part["close"].pct_change()
        part["stock_volume_change_1"] = part["volume"].pct_change()
        part["stock_turnover_change_1"] = part["turnover_rate"].pct_change()
        part["limit_up_flag"] = (part["pct_change"] >= 9.5).astype(float)
        rows.append(part)
    panel = pd.concat(rows, ignore_index=True)
    daily = (
        panel.groupby("date", as_index=False)
        .agg(
            power_return_1=("stock_return_1", "mean"),
            power_up_ratio=("stock_return_1", lambda s: float(np.nanmean((s > 0).astype(float)))),
            power_limit_up_ratio=("limit_up_flag", "mean"),
            power_volume_change_1=("stock_volume_change_1", "mean"),
            power_turnover_change_1=("stock_turnover_change_1", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["power_index_close"] = (1 + daily["power_return_1"].fillna(0.0)).cumprod() * 100
    daily["power_return_5"] = daily["power_index_close"].pct_change(5)
    daily["power_volatility_10"] = daily["power_return_1"].rolling(10, min_periods=10).std()
    return daily


def fetch_market_data(
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START_DATE,
    end: str | None = None,
    force_refresh: bool = False,
) -> MarketData:
    pool_symbols = list(dict.fromkeys([symbol, *POWER_STOCK_POOL.keys()]))
    stocks = {
        stock_symbol: fetch_stock_history(symbol=stock_symbol, start_date=start, end_date=end, force_refresh=force_refresh)
        for stock_symbol in pool_symbols
    }
    indices = {
        index_symbol: fetch_index_history(index_symbol, start_date=start, end_date=end, force_refresh=force_refresh)
        for index_symbol in INDEX_DISPLAY_NAMES
    }
    northbound = fetch_northbound_history(start_date=start, end_date=end, force_refresh=force_refresh)
    trade_calendar = fetch_trade_calendar(force_refresh=force_refresh)
    industry_proxy = build_power_industry_proxy(stocks)
    return MarketData(
        target_stock=stocks[symbol],
        stocks=stocks,
        indices=indices,
        northbound=northbound,
        industry_proxy=industry_proxy,
        trade_calendar=trade_calendar,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and cache A-share demo data.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force-refresh", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    market_data = fetch_market_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        force_refresh=args.force_refresh,
    )
    print(f"Fetched {len(market_data.stocks)} stock series for pooled training.")
    print(f"Target series rows: {len(market_data.target_stock)}")
    for index_symbol, frame in market_data.indices.items():
        print(f"Fetched {len(frame)} rows for {index_symbol}.")
    print(f"Northbound rows: {len(market_data.northbound)}")
    print(f"Industry proxy rows: {len(market_data.industry_proxy)}")
    print(f"Trade calendar rows: {len(market_data.trade_calendar)}")


if __name__ == "__main__":
    main()
