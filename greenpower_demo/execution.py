from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PaperPosition:
    symbol: str
    shares: int
    price: float


@dataclass(slots=True)
class PaperAccountState:
    generated_at: str
    mode: str
    cash: float
    equity: float
    market_value: float
    positions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrokerAdapter:
    mode = "review"

    def get_account_state(self) -> dict[str, Any]:  # pragma: no cover - interface only
        raise NotImplementedError

    def submit_orders(self, orders: list[dict[str, Any]]) -> dict[str, Any]:  # pragma: no cover - interface only
        raise NotImplementedError


class PaperBrokerAdapter(BrokerAdapter):
    mode = "paper"

    def __init__(self, path: Path, initial_cash: float) -> None:
        self.path = path
        self.initial_cash = initial_cash

    def get_account_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "mode": self.mode,
                "cash": float(self.initial_cash),
                "equity": float(self.initial_cash),
                "market_value": 0.0,
                "positions": [],
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def submit_orders(self, orders: list[dict[str, Any]]) -> dict[str, Any]:
        return {"submitted": False, "review_required": True, "orders": orders}


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _normalize_positions(raw_positions: list[dict[str, Any]]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for position in raw_positions:
        symbol = str(position.get("symbol", "")).zfill(6)
        shares = int(position.get("shares", 0))
        if symbol:
            normalized[symbol] = shares
    return normalized


def rebalance_paper_account(
    account_path: Path,
    latest_portfolio: pd.DataFrame,
    price_frame: pd.DataFrame,
    initial_cash: float,
    lot_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    broker = PaperBrokerAdapter(account_path, initial_cash=initial_cash)
    current = broker.get_account_state()
    current_positions = _normalize_positions(current.get("positions", []))
    cash = float(current.get("cash", initial_cash))

    latest_prices = (
        price_frame[["symbol", "close"]]
        .sort_values("symbol")
        .drop_duplicates(subset=["symbol"], keep="last")
        .set_index("symbol")["close"]
        .to_dict()
    )

    current_market_value = sum(current_positions.get(symbol, 0) * float(latest_prices.get(symbol, 0.0)) for symbol in current_positions)
    equity = cash + current_market_value
    if equity <= 0:
        equity = initial_cash
        cash = initial_cash

    desired_positions: dict[str, int] = {}
    for row in latest_portfolio.itertuples(index=False):
        price = float(latest_prices.get(row.symbol, 0.0))
        if price <= 0:
            continue
        target_weight = float(getattr(row, "target_weight", 0.0))
        target_value = equity * target_weight
        target_shares = int(target_value // (price * lot_size)) * lot_size
        desired_positions[row.symbol] = max(target_shares, 0)

    orders: list[dict[str, Any]] = []
    all_symbols = set(current_positions) | set(desired_positions)
    for symbol in sorted(all_symbols):
        current_shares = current_positions.get(symbol, 0)
        target_shares = desired_positions.get(symbol, 0)
        delta = target_shares - current_shares
        if delta == 0:
            continue
        price = float(latest_prices.get(symbol, 0.0))
        side = "BUY" if delta > 0 else "SELL"
        orders.append(
            {
                "symbol": symbol,
                "side": side,
                "shares": abs(delta),
                "price": price,
                "estimated_notional": abs(delta) * price,
                "review_required": True,
            }
        )

    new_positions = []
    for symbol, shares in desired_positions.items():
        price = float(latest_prices.get(symbol, 0.0))
        new_positions.append({"symbol": symbol, "shares": shares, "price": price})

    market_value = sum(position["shares"] * position["price"] for position in new_positions)
    realized_cash = float(equity - market_value)
    account_state = PaperAccountState(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        mode="paper",
        cash=float(realized_cash),
        equity=float(realized_cash + market_value),
        market_value=float(market_value),
        positions=new_positions,
    ).to_dict()
    account_path.write_text(json.dumps(account_state, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return account_state, orders


def build_ai_trade_context(
    summary: dict[str, Any],
    metrics: dict[str, Any],
    trade_plan: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "paper-only",
        "review_required": True,
        "live_trading_enabled": False,
        "reason": "Broker credentials and explicit risk approval are required before any live execution.",
        "strategy_summary": summary,
        "metrics": metrics,
        "trade_plan": trade_plan,
        "risk_limits": {
            "max_single_weight": config.get("max_single_weight"),
            "max_turnover": config.get("max_turnover"),
            "max_positions_per_industry": config.get("max_positions_per_industry"),
        },
    }

