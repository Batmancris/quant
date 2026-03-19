from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

DEFAULT_SYMBOL = "000537"
DISPLAY_NAME = "绿发电力"
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_TARGET_DATE: str | None = None
DEFAULT_HORIZON_DAYS = 5
DEFAULT_RETURN_THRESHOLD = 0.01
DEFAULT_STRATEGY_ID = "power_multi_factor_strategy"

POWER_STOCK_POOL = {
    "000027": "深圳能源",
    "000537": "绿发电力",
    "000539": "粤电力A",
    "000543": "皖能电力",
    "000600": "建投能源",
    "000875": "吉电股份",
    "000883": "湖北能源",
    "600011": "华能国际",
    "600021": "上海电力",
    "600023": "浙能电力",
    "600025": "华能水电",
    "600027": "华电国际",
    "600795": "国电电力",
    "600900": "长江电力",
    "600905": "三峡能源",
    "601991": "大唐发电",
}

INDEX_DISPLAY_NAMES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sh000300": "沪深300",
}
INDEX_PREFIXES = {
    "sh000001": "sse",
    "sz399001": "szse",
    "sh000300": "hs300",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
RAW_CACHE_DIR = CACHE_DIR / "raw"
FEATURE_CACHE_DIR = CACHE_DIR / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
STATIC_SITE_DIR = PROJECT_ROOT / "docs"
STATIC_SITE_DATA_PATH = STATIC_SITE_DIR / "site-data.json"


@dataclass(slots=True)
class TrainingConfig:
    prediction_horizon_days: int = DEFAULT_HORIZON_DAYS
    return_threshold: float = DEFAULT_RETURN_THRESHOLD
    verbose: bool = True

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


@dataclass(slots=True)
class QuantStrategyConfig:
    strategy_id: str = DEFAULT_STRATEGY_ID
    holding_period_days: int = 5
    rebalance_frequency_days: int = 5
    top_k: int = 3
    train_ratio: float = 0.70
    transaction_cost_bps: float = 10.0
    sell_tax_bps: float = 5.0
    min_cross_section: int = 5
    seed: int = 42

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def ensure_directories() -> None:
    for path in (CACHE_DIR, RAW_CACHE_DIR, FEATURE_CACHE_DIR, ARTIFACTS_DIR, STATIC_SITE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _date_token(value: str | None) -> str:
    return (value or date.today().isoformat()).replace("-", "")


def raw_cache_path(kind: str, symbol: str, start_date: str, end_date: str | None) -> Path:
    ensure_directories()
    return RAW_CACHE_DIR / f"{kind}_{symbol}_{_date_token(start_date)}_{_date_token(end_date)}.csv"


def feature_cache_path(symbol: str, start_date: str, end_date: str | None) -> Path:
    ensure_directories()
    return FEATURE_CACHE_DIR / f"features_{symbol}_{_date_token(start_date)}_{_date_token(end_date)}.csv"


def trade_calendar_cache_path() -> Path:
    ensure_directories()
    return RAW_CACHE_DIR / "trade_calendar.csv"


def artifact_dir(symbol: str) -> Path:
    ensure_directories()
    target = ARTIFACTS_DIR / symbol
    target.mkdir(parents=True, exist_ok=True)
    return target
