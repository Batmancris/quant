from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import STATIC_SITE_DATA_PATH, STATIC_SITE_DIR


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


def export_static_site(dashboard: dict[str, Any]) -> str:
    STATIC_SITE_DIR.mkdir(parents=True, exist_ok=True)
    summary = dashboard["run_summary"]
    feature_frame = dashboard["feature_frame"].copy()
    latest_portfolio = dashboard["latest_portfolio"].copy()
    target_symbol = summary["symbol"]
    price_history = (
        feature_frame.loc[feature_frame["symbol"] == target_symbol, ["date", "close", "volume"]]
        .sort_values("date")
        .tail(320)
        .reset_index(drop=True)
    )
    payload = {
        "summary": summary,
        "metrics": dashboard["metrics"],
        "price_history": price_history.to_dict(orient="records"),
        "backtest_daily": dashboard["backtest_daily"].sort_values("date").to_dict(orient="records"),
        "factor_ic_history": dashboard["factor_ic_history"].to_dict(orient="records"),
        "factor_weights": dashboard["factor_weights"].to_dict(orient="records"),
        "rebalance_log": dashboard["rebalance_log"].sort_values("signal_date", ascending=False).head(60).to_dict(orient="records"),
        "latest_portfolio": latest_portfolio.to_dict(orient="records"),
        "stock_pool": summary.get("stock_pool", []),
        "artifact_dir": dashboard["artifact_dir"],
    }
    STATIC_SITE_DATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return str(STATIC_SITE_DATA_PATH)
