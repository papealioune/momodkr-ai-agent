from __future__ import annotations

ALLOWED_ASSETS = ["BTC", "ETH", "SOL"]

BINANCE_SYMBOL_MAP: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

HYPERLIQUID_SYMBOL_MAP: dict[str, str] = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
}

REVERSE_BINANCE_MAP: dict[str, str] = {v: k for k, v in BINANCE_SYMBOL_MAP.items()}

BINANCE_LISTING_DATES: dict[str, str] = {
    "BTCUSDT": "2019-09-10",
    "ETHUSDT": "2019-11-27",
    "SOLUSDT": "2021-07-12",
}
