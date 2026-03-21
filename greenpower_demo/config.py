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
DEFAULT_INITIAL_CAPITAL = 1_000_000.0

CURATED_UNIVERSE = {
    "000027": {"name": "深圳能源", "industry": "公用事业"},
    "000333": {"name": "美的集团", "industry": "家电"},
    "000537": {"name": "绿发电力", "industry": "公用事业"},
    "000539": {"name": "粤电力A", "industry": "公用事业"},
    "000543": {"name": "皖能电力", "industry": "公用事业"},
    "000600": {"name": "建投能源", "industry": "公用事业"},
    "000651": {"name": "格力电器", "industry": "家电"},
    "000858": {"name": "五粮液", "industry": "食品饮料"},
    "000875": {"name": "吉电股份", "industry": "公用事业"},
    "000883": {"name": "湖北能源", "industry": "公用事业"},
    "000938": {"name": "紫光股份", "industry": "计算机"},
    "000963": {"name": "华东医药", "industry": "医药生物"},
    "002371": {"name": "北方华创", "industry": "电子"},
    "002415": {"name": "海康威视", "industry": "电子"},
    "002594": {"name": "比亚迪", "industry": "汽车"},
    "002714": {"name": "牧原股份", "industry": "农林牧渔"},
    "300014": {"name": "亿纬锂能", "industry": "电力设备"},
    "300750": {"name": "宁德时代", "industry": "电力设备"},
    "300760": {"name": "迈瑞医疗", "industry": "医药生物"},
    "600009": {"name": "上海机场", "industry": "交通运输"},
    "600011": {"name": "华能国际", "industry": "公用事业"},
    "600021": {"name": "上海电力", "industry": "公用事业"},
    "600023": {"name": "浙能电力", "industry": "公用事业"},
    "600025": {"name": "华能水电", "industry": "公用事业"},
    "600027": {"name": "华电国际", "industry": "公用事业"},
    "600028": {"name": "中国石化", "industry": "石油石化"},
    "600030": {"name": "中信证券", "industry": "非银金融"},
    "600031": {"name": "三一重工", "industry": "机械设备"},
    "600050": {"name": "中国联通", "industry": "通信"},
    "600276": {"name": "恒瑞医药", "industry": "医药生物"},
    "600309": {"name": "万华化学", "industry": "基础化工"},
    "600519": {"name": "贵州茅台", "industry": "食品饮料"},
    "600795": {"name": "国电电力", "industry": "公用事业"},
    "600809": {"name": "山西汾酒", "industry": "食品饮料"},
    "600887": {"name": "伊利股份", "industry": "食品饮料"},
    "600900": {"name": "长江电力", "industry": "公用事业"},
    "600905": {"name": "三峡能源", "industry": "公用事业"},
    "600938": {"name": "中国海油", "industry": "石油石化"},
    "600941": {"name": "中国移动", "industry": "通信"},
    "601006": {"name": "大秦铁路", "industry": "交通运输"},
    "601088": {"name": "中国神华", "industry": "煤炭"},
    "601166": {"name": "兴业银行", "industry": "银行"},
    "601211": {"name": "国泰君安", "industry": "非银金融"},
    "601288": {"name": "农业银行", "industry": "银行"},
    "601318": {"name": "中国平安", "industry": "非银金融"},
    "601390": {"name": "中国中铁", "industry": "建筑装饰"},
    "601398": {"name": "工商银行", "industry": "银行"},
    "601601": {"name": "中国太保", "industry": "非银金融"},
    "601628": {"name": "中国人寿", "industry": "非银金融"},
    "601668": {"name": "中国建筑", "industry": "建筑装饰"},
    "601669": {"name": "中国电建", "industry": "建筑装饰"},
    "601688": {"name": "华泰证券", "industry": "非银金融"},
    "601728": {"name": "中国电信", "industry": "通信"},
    "601766": {"name": "中国中车", "industry": "机械设备"},
    "601816": {"name": "京沪高铁", "industry": "交通运输"},
    "601857": {"name": "中国石油", "industry": "石油石化"},
    "601899": {"name": "紫金矿业", "industry": "有色金属"},
    "601919": {"name": "中远海控", "industry": "交通运输"},
    "601939": {"name": "建设银行", "industry": "银行"},
    "601991": {"name": "大唐发电", "industry": "公用事业"},
    "603986": {"name": "兆易创新", "industry": "电子"},
    "688041": {"name": "海光信息", "industry": "电子"},
    "688981": {"name": "中芯国际", "industry": "电子"},
}

POWER_STOCK_POOL = {
    symbol: info["name"]
    for symbol, info in CURATED_UNIVERSE.items()
    if info["industry"] == "公用事业"
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
FUNDAMENTAL_CACHE_DIR = CACHE_DIR / "fundamentals"
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
    top_k: int = 8
    train_ratio: float = 0.70
    initial_train_days: int = 756
    walk_forward_test_days: int = 63
    walk_forward_step_days: int = 63
    universe_size: int = 72
    universe_per_industry_cap: int = 4
    max_positions_per_industry: int = 2
    max_single_weight: float = 0.18
    max_turnover: float = 0.60
    min_market_cap_quantile: float = 0.20
    transaction_cost_bps: float = 10.0
    sell_tax_bps: float = 5.0
    min_cross_section: int = 8
    initial_cash: float = DEFAULT_INITIAL_CAPITAL
    lot_size: int = 100
    broker_mode: str = "paper"
    seed: int = 42

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def ensure_directories() -> None:
    for path in (
        CACHE_DIR,
        RAW_CACHE_DIR,
        FEATURE_CACHE_DIR,
        FUNDAMENTAL_CACHE_DIR,
        ARTIFACTS_DIR,
        STATIC_SITE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _date_token(value: str | None) -> str:
    return (value or date.today().isoformat()).replace("-", "")


def raw_cache_path(kind: str, symbol: str, start_date: str, end_date: str | None) -> Path:
    ensure_directories()
    return RAW_CACHE_DIR / f"{kind}_{symbol}_{_date_token(start_date)}_{_date_token(end_date)}.csv"


def feature_cache_path(symbol: str, start_date: str, end_date: str | None) -> Path:
    ensure_directories()
    return FEATURE_CACHE_DIR / f"features_{symbol}_{_date_token(start_date)}_{_date_token(end_date)}.csv"


def fundamental_cache_path(kind: str, token: str) -> Path:
    ensure_directories()
    return FUNDAMENTAL_CACHE_DIR / f"{kind}_{token}.csv"


def trade_calendar_cache_path() -> Path:
    ensure_directories()
    return RAW_CACHE_DIR / "trade_calendar.csv"


def artifact_dir(symbol: str) -> Path:
    ensure_directories()
    target = ARTIFACTS_DIR / symbol
    target.mkdir(parents=True, exist_ok=True)
    return target
