from datetime import date

from data.collectors.asset_config import (
    ALLOWED_ASSETS,
    BINANCE_LISTING_DATES,
    BINANCE_SYMBOL_MAP,
    REVERSE_BINANCE_MAP,
)
from data.collectors.binance_vision_l2_collector import (
    BINANCE_VISION_DAILY,
    STREAMS,
    FetchTask,
    build_tasks,
    day_range,
    filter_to_listing,
)


def test_streams_are_the_three_we_need() -> None:
    assert set(STREAMS) == {"bookTicker", "aggTrades", "bookDepth"}


def test_url_targets_futures_um_not_spot() -> None:
    assert BINANCE_VISION_DAILY == "https://data.binance.vision/data/futures/um/daily"


def test_asset_config_v1_universe() -> None:
    assert ALLOWED_ASSETS == ["BTC", "ETH", "SOL"]
    assert BINANCE_SYMBOL_MAP["BTC"] == "BTCUSDT"
    assert REVERSE_BINANCE_MAP["ETHUSDT"] == "ETH"


def test_fetch_task_url_shape() -> None:
    t = FetchTask(symbol="BTCUSDT", stream="bookTicker", day=date(2024, 1, 15))
    assert t.filename == "BTCUSDT-bookTicker-2024-01-15.zip"
    assert t.url == (
        "https://data.binance.vision/data/futures/um/daily/bookTicker/"
        "BTCUSDT/BTCUSDT-bookTicker-2024-01-15.zip"
    )


def test_day_range_inclusive() -> None:
    days = day_range(date(2024, 1, 1), date(2024, 1, 3))
    assert days == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


def test_day_range_rejects_inverted() -> None:
    import pytest

    with pytest.raises(ValueError):
        day_range(date(2024, 1, 5), date(2024, 1, 1))


def test_filter_to_listing_drops_pre_listing_days() -> None:
    listing = BINANCE_LISTING_DATES["SOLUSDT"]
    assert listing == "2021-07-12"
    days = [date(2021, 7, 10), date(2021, 7, 12), date(2021, 7, 15)]
    kept = filter_to_listing("SOLUSDT", days)
    assert kept == [date(2021, 7, 12), date(2021, 7, 15)]


def test_build_tasks_covers_all_symbol_stream_day_combinations() -> None:
    tasks = build_tasks(
        ["BTCUSDT", "ETHUSDT"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
    )
    assert len(tasks) == 2 * 2 * 3  # 2 symbols x 2 days x 3 streams
    symbols = {t.symbol for t in tasks}
    streams = {t.stream for t in tasks}
    assert symbols == {"BTCUSDT", "ETHUSDT"}
    assert streams == set(STREAMS)
