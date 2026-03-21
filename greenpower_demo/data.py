from __future__ import annotations

import argparse
import re
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
        CURATED_UNIVERSE,
        DEFAULT_START_DATE,
        DEFAULT_SYMBOL,
        INDEX_DISPLAY_NAMES,
        POWER_STOCK_POOL,
        QuantStrategyConfig,
        fundamental_cache_path,
        raw_cache_path,
        trade_calendar_cache_path,
    )
else:
    from .config import (
        CURATED_UNIVERSE,
        DEFAULT_START_DATE,
        DEFAULT_SYMBOL,
        INDEX_DISPLAY_NAMES,
        POWER_STOCK_POOL,
        QuantStrategyConfig,
        fundamental_cache_path,
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
    universe_metadata: pd.DataFrame
    fundamental_history: pd.DataFrame
    dividend_history: pd.DataFrame
    latest_fundamentals: pd.DataFrame


def _retry_fetch(fetcher: Callable[[], pd.DataFrame], description: str, retries: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fetcher()
        except Exception as exc:  # pragma: no cover - live path
            last_error = exc
            if attempt == retries:
                break
            time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to fetch {description}") from last_error


def _load_or_fetch_csv(
    path: Path,
    fetcher: Callable[[], pd.DataFrame],
    description: str,
    force_refresh: bool,
    date_columns: list[str] | None = None,
    allow_empty_on_failure: bool = False,
) -> pd.DataFrame:
    if path.exists() and not force_refresh:
        return pd.read_csv(path, parse_dates=date_columns or [])
    try:
        frame = _retry_fetch(fetcher, description=description)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return frame
    except Exception as exc:
        if path.exists():
            warnings.warn(f"{description} refresh failed, using cached data: {exc}", stacklevel=2)
            return pd.read_csv(path, parse_dates=date_columns or [])
        if allow_empty_on_failure:
            warnings.warn(f"{description} unavailable, continuing without it: {exc}", stacklevel=2)
            return pd.DataFrame()
        raise


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
    for column in [
        "north_net_deal",
        "north_buy_amount",
        "north_sell_amount",
        "north_cumulative",
        "north_inflow",
        "north_balance",
        "north_hold_value",
        "hs300_close_ref",
        "hs300_pct_ref",
    ]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _normalize_trade_calendar(frame: pd.DataFrame) -> pd.DatetimeIndex:
    trade_dates = pd.to_datetime(frame["trade_date"]).sort_values().drop_duplicates()
    return pd.DatetimeIndex(trade_dates)


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

    return _load_or_fetch_csv(cache_path, _fetch, f"stock history for {symbol}", force_refresh, ["date"])


def fetch_index_history(symbol: str, start_date: str, end_date: str | None = None, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = raw_cache_path("index", symbol, start_date, end_date)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date or date.today().isoformat())

    def _fetch() -> pd.DataFrame:
        frame = ak.stock_zh_index_daily(symbol=symbol)
        normalized = _normalize_index_history(frame, symbol)
        mask = (normalized["date"] >= start_ts) & (normalized["date"] <= end_ts)
        return normalized.loc[mask].reset_index(drop=True)

    return _load_or_fetch_csv(cache_path, _fetch, f"index history for {symbol}", force_refresh, ["date"])


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

    frame = _load_or_fetch_csv(cache_path, _fetch, "northbound history", force_refresh, ["date"], allow_empty_on_failure=True)
    if frame.empty:
        return _normalize_northbound_history(frame)
    return frame


def fetch_trade_calendar(force_refresh: bool = False) -> pd.DatetimeIndex:
    cache_path = trade_calendar_cache_path()

    def _fetch() -> pd.DataFrame:
        return ak.tool_trade_date_hist_sina()

    frame = _load_or_fetch_csv(cache_path, _fetch, "A-share trade calendar", force_refresh, ["trade_date"])
    return _normalize_trade_calendar(frame)


def _quarter_candidates() -> list[str]:
    today = pd.Timestamp(date.today().isoformat())
    candidates: list[str] = []
    current = today.to_period("Q")
    for offset in range(0, 12):
        quarter = current - offset
        candidates.append(quarter.end_time.strftime("%Y%m%d"))
    return list(dict.fromkeys(candidates))


def _normalize_fundamental_report(frame: pd.DataFrame, report_date: str, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "industry",
                "report_date",
                "announcement_date",
                "eps",
                "revenue",
                "net_profit",
                "bvps",
                "roe",
                "operating_cashflow_per_share",
                "gross_margin",
                "report_source",
            ]
        )
    renamed = frame.rename(
        columns={
            "股票代码": "symbol",
            "股票简称": "name",
            "所处行业": "industry",
            "每股收益": "eps",
            "营业总收入-营业总收入": "revenue",
            "营业收入-营业收入": "revenue",
            "净利润-净利润": "net_profit",
            "每股净资产": "bvps",
            "净资产收益率": "roe",
            "每股经营现金流量": "operating_cashflow_per_share",
            "销售毛利率": "gross_margin",
            "最新公告日期": "announcement_date",
            "公告日期": "announcement_date",
        }
    ).copy()
    renamed["report_date"] = pd.to_datetime(report_date)
    renamed["announcement_date"] = pd.to_datetime(renamed.get("announcement_date"), errors="coerce")
    renamed["industry"] = renamed.get("industry", pd.Series(index=renamed.index, dtype=object)).fillna("未知行业")
    for column in [
        "eps",
        "revenue",
        "net_profit",
        "bvps",
        "roe",
        "operating_cashflow_per_share",
        "gross_margin",
    ]:
        renamed[column] = pd.to_numeric(renamed.get(column), errors="coerce")
    renamed["report_source"] = source
    return renamed[
        [
            "symbol",
            "name",
            "industry",
            "report_date",
            "announcement_date",
            "eps",
            "revenue",
            "net_profit",
            "bvps",
            "roe",
            "operating_cashflow_per_share",
            "gross_margin",
            "report_source",
        ]
    ].dropna(subset=["symbol"])


def fetch_report_snapshot(report_date: str, source: str = "yjbb", force_refresh: bool = False) -> pd.DataFrame:
    cache_path = fundamental_cache_path(source, report_date)

    def _fetch() -> pd.DataFrame:
        if source == "yjbb":
            frame = ak.stock_yjbb_em(date=report_date)
        elif source == "yjkb":
            frame = ak.stock_yjkb_em(date=report_date)
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"Unsupported report source: {source}")
        return _normalize_fundamental_report(frame, report_date=report_date, source=source)

    return _load_or_fetch_csv(
        cache_path,
        _fetch,
        f"fundamental report {source} {report_date}",
        force_refresh,
        ["report_date", "announcement_date"],
        allow_empty_on_failure=True,
    )


def fetch_latest_available_fundamentals(force_refresh: bool = False) -> tuple[pd.DataFrame, str | None, str | None]:
    for report_date in _quarter_candidates():
        for source in ("yjbb", "yjkb"):
            frame = fetch_report_snapshot(report_date=report_date, source=source, force_refresh=force_refresh)
            if not frame.empty:
                return frame, report_date, source
    return pd.DataFrame(), None, None


def _select_dynamic_universe(
    target_symbol: str,
    latest_fundamentals: pd.DataFrame,
    config: QuantStrategyConfig,
) -> pd.DataFrame:
    selected_rows: list[dict[str, str]] = []
    if latest_fundamentals.empty:
        return pd.DataFrame(columns=["symbol", "name", "industry", "source"])

    frame = latest_fundamentals.copy()
    frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce")
    frame["roe"] = pd.to_numeric(frame["roe"], errors="coerce")
    frame["industry"] = frame["industry"].fillna("未知行业")
    frame["name"] = frame["name"].fillna(frame["symbol"])
    frame = frame.sort_values(["revenue", "roe"], ascending=[False, False]).reset_index(drop=True)

    industry_counts: dict[str, int] = {}
    selected_symbols: set[str] = set()
    for row in frame.itertuples(index=False):
        if len(selected_rows) >= config.universe_size:
            break
        industry = str(row.industry or "未知行业")
        if industry_counts.get(industry, 0) >= config.universe_per_industry_cap:
            continue
        symbol = str(row.symbol).zfill(6)
        if symbol in selected_symbols:
            continue
        selected_symbols.add(symbol)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        selected_rows.append({"symbol": symbol, "name": row.name, "industry": industry, "source": "dynamic"})

    if target_symbol not in selected_symbols:
        fallback = latest_fundamentals.loc[latest_fundamentals["symbol"] == target_symbol]
        if not fallback.empty:
            row = fallback.iloc[0]
            selected_rows.append(
                {
                    "symbol": target_symbol,
                    "name": row.get("name", target_symbol),
                    "industry": row.get("industry", CURATED_UNIVERSE.get(target_symbol, {}).get("industry", "未知行业")),
                    "source": "target-override",
                }
            )
            selected_symbols.add(target_symbol)

    return pd.DataFrame(selected_rows)


def build_universe_metadata(target_symbol: str, config: QuantStrategyConfig, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, str | None, str | None]:
    latest_fundamentals, latest_report_date, latest_report_source = fetch_latest_available_fundamentals(force_refresh=force_refresh)
    dynamic = _select_dynamic_universe(target_symbol=target_symbol, latest_fundamentals=latest_fundamentals, config=config)

    rows = dynamic.to_dict(orient="records")
    selected_symbols = {row["symbol"] for row in rows}
    for symbol, info in CURATED_UNIVERSE.items():
        if len(rows) >= config.universe_size:
            break
        if symbol in selected_symbols:
            continue
        rows.append({"symbol": symbol, "name": info["name"], "industry": info["industry"], "source": "fallback"})
        selected_symbols.add(symbol)

    if target_symbol not in selected_symbols:
        info = CURATED_UNIVERSE.get(target_symbol, {"name": target_symbol, "industry": "公用事业"})
        rows.append({"symbol": target_symbol, "name": info["name"], "industry": info["industry"], "source": "target-override"})

    metadata = pd.DataFrame(rows).drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    metadata["symbol"] = metadata["symbol"].astype(str).str.zfill(6)
    return metadata, latest_fundamentals, latest_report_date, latest_report_source


def build_fundamental_history(
    universe_metadata: pd.DataFrame,
    start_date: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    universe_symbols = set(universe_metadata["symbol"].astype(str))
    start_year = max(2014, pd.Timestamp(start_date).year - 1)
    end_year = date.today().year - 1
    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        report_date = f"{year}1231"
        frame = fetch_report_snapshot(report_date=report_date, source="yjbb", force_refresh=force_refresh)
        if frame.empty:
            continue
        filtered = frame.loc[frame["symbol"].astype(str).isin(universe_symbols)].copy()
        if not filtered.empty:
            frames.append(filtered)

    latest_fundamentals, latest_report_date, latest_report_source = fetch_latest_available_fundamentals(force_refresh=force_refresh)
    if latest_report_date is not None and latest_report_source is not None and not latest_fundamentals.empty:
        latest_filtered = latest_fundamentals.loc[latest_fundamentals["symbol"].astype(str).isin(universe_symbols)].copy()
        if not latest_filtered.empty:
            frames.append(latest_filtered)

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "industry",
                "report_date",
                "announcement_date",
                "eps",
                "revenue",
                "net_profit",
                "bvps",
                "roe",
                "operating_cashflow_per_share",
                "gross_margin",
                "report_source",
                "cashflow_quality",
                "fundamental_asof_date",
            ]
        )

    history = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol", "report_date"], keep="last")
    history = history.merge(universe_metadata[["symbol", "industry"]], on="symbol", how="left", suffixes=("", "_fallback"))
    history["industry"] = history["industry"].fillna(history["industry_fallback"]).fillna("未知行业")
    history.drop(columns=["industry_fallback"], inplace=True)
    history["cashflow_quality"] = history["operating_cashflow_per_share"] / history["eps"].abs().replace(0, np.nan)
    history["fundamental_asof_date"] = history["announcement_date"].fillna(history["report_date"] + pd.offsets.Day(120))
    history = history.sort_values(["symbol", "fundamental_asof_date", "report_date"]).reset_index(drop=True)
    return history


def _parse_dividend_per_share(value: object, description: object) -> float | None:
    ratio = pd.to_numeric(value, errors="coerce")
    if pd.notna(ratio):
        return float(ratio) / 10.0
    if isinstance(description, str):
        match = re.search(r"10派([0-9]+(?:\.[0-9]+)?)元", description)
        if match:
            return float(match.group(1)) / 10.0
    return None


def fetch_dividend_history(symbol: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = fundamental_cache_path("dividend", symbol)

    def _fetch() -> pd.DataFrame:
        frame = ak.stock_dividend_cninfo(symbol=symbol)
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "dividend_announce_date", "dividend_per_share", "report_label"])
        normalized = frame.rename(
            columns={
                "实施方案公告日期": "dividend_announce_date",
                "派息比例": "cash_dividend_ratio",
                "实施方案分红说明": "dividend_description",
                "报告时间": "report_label",
            }
        ).copy()
        normalized["symbol"] = symbol
        normalized["dividend_announce_date"] = pd.to_datetime(normalized["dividend_announce_date"], errors="coerce")
        normalized["dividend_per_share"] = [
            _parse_dividend_per_share(value, description)
            for value, description in zip(normalized.get("cash_dividend_ratio", []), normalized.get("dividend_description", []))
        ]
        normalized["dividend_per_share"] = pd.to_numeric(normalized["dividend_per_share"], errors="coerce")
        return normalized[["symbol", "dividend_announce_date", "dividend_per_share", "report_label"]].dropna(subset=["dividend_announce_date"])

    return _load_or_fetch_csv(
        cache_path,
        _fetch,
        f"dividend history for {symbol}",
        force_refresh,
        ["dividend_announce_date"],
        allow_empty_on_failure=True,
    )


def build_dividend_history(universe_metadata: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    frames = [fetch_dividend_history(symbol=symbol, force_refresh=force_refresh) for symbol in universe_metadata["symbol"].astype(str)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["symbol", "dividend_announce_date", "dividend_per_share", "report_label"])
    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(["symbol", "dividend_announce_date"]).reset_index(drop=True)
    return history


def build_industry_proxy(stocks: dict[str, pd.DataFrame], universe_metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    industry_map = universe_metadata.set_index("symbol")["industry"].to_dict()
    for symbol, frame in stocks.items():
        part = frame[["date", "close", "volume", "amount", "turnover_rate", "pct_change"]].copy()
        part["symbol"] = symbol
        part["industry"] = industry_map.get(symbol, "未知行业")
        part["stock_return_1"] = part["close"].pct_change()
        part["stock_volume_change_1"] = part["volume"].pct_change()
        part["stock_turnover_change_1"] = part["turnover_rate"].pct_change()
        part["limit_up_flag"] = (part["pct_change"] >= 9.5).astype(float)
        rows.append(part)
    panel = pd.concat(rows, ignore_index=True)
    daily = (
        panel.groupby("date", as_index=False)
        .agg(
            universe_return_1=("stock_return_1", "mean"),
            universe_up_ratio=("stock_return_1", lambda s: float(np.nanmean((s > 0).astype(float)))),
            universe_limit_up_ratio=("limit_up_flag", "mean"),
            universe_volume_change_1=("stock_volume_change_1", "mean"),
            universe_turnover_change_1=("stock_turnover_change_1", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["universe_index_close"] = (1 + daily["universe_return_1"].fillna(0.0)).cumprod() * 100
    daily["universe_return_5"] = daily["universe_index_close"].pct_change(5)
    daily["universe_volatility_10"] = daily["universe_return_1"].rolling(10, min_periods=10).std()
    return daily


def fetch_market_data(
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START_DATE,
    end: str | None = None,
    force_refresh: bool = False,
    config: QuantStrategyConfig | None = None,
) -> MarketData:
    config = config or QuantStrategyConfig()
    universe_metadata, latest_fundamentals, _, _ = build_universe_metadata(symbol, config=config, force_refresh=force_refresh)
    symbols = list(dict.fromkeys([symbol, *universe_metadata["symbol"].astype(str).tolist()]))

    stocks = {
        stock_symbol: fetch_stock_history(symbol=stock_symbol, start_date=start, end_date=end, force_refresh=force_refresh)
        for stock_symbol in symbols
    }
    indices = {
        index_symbol: fetch_index_history(index_symbol, start_date=start, end_date=end, force_refresh=force_refresh)
        for index_symbol in INDEX_DISPLAY_NAMES
    }
    northbound = fetch_northbound_history(start_date=start, end_date=end, force_refresh=force_refresh)
    trade_calendar = fetch_trade_calendar(force_refresh=force_refresh)
    fundamental_history = build_fundamental_history(universe_metadata, start_date=start, force_refresh=force_refresh)
    dividend_history = build_dividend_history(universe_metadata, force_refresh=force_refresh)
    industry_proxy = build_industry_proxy(stocks, universe_metadata)

    return MarketData(
        target_stock=stocks[symbol],
        stocks=stocks,
        indices=indices,
        northbound=northbound,
        industry_proxy=industry_proxy,
        trade_calendar=trade_calendar,
        universe_metadata=universe_metadata,
        fundamental_history=fundamental_history,
        dividend_history=dividend_history,
        latest_fundamentals=latest_fundamentals,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and cache A-share demo data.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--universe-size", type=int, default=72)
    parser.add_argument("--universe-per-industry-cap", type=int, default=4)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = QuantStrategyConfig(universe_size=args.universe_size, universe_per_industry_cap=args.universe_per_industry_cap)
    market_data = fetch_market_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        force_refresh=args.force_refresh,
        config=config,
    )
    print(f"Fetched {len(market_data.stocks)} stock series for pooled training.")
    print(f"Target series rows: {len(market_data.target_stock)}")
    print(f"Universe symbols: {len(market_data.universe_metadata)}")
    print(f"Fundamental rows: {len(market_data.fundamental_history)}")
    print(f"Dividend rows: {len(market_data.dividend_history)}")
    for index_symbol, frame in market_data.indices.items():
        print(f"Fetched {len(frame)} rows for {index_symbol}.")
    print(f"Northbound rows: {len(market_data.northbound)}")
    print(f"Industry proxy rows: {len(market_data.industry_proxy)}")
    print(f"Trade calendar rows: {len(market_data.trade_calendar)}")


if __name__ == "__main__":
    main()
