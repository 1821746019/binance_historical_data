import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from binance_historical_data import (
    detect_kline_gaps,
    fetch_klines_range,
    fill_kline_gaps,
    interval_to_milliseconds,
    patch_kline_file,
    patch_pyarrow_kline_table,
)


@pytest.mark.parametrize(
    "interval_str,expected_ms",
    [
        ("1s", 1_000),
        ("1m", 60_000),
        ("3m", 180_000),
        ("5m", 300_000),
        ("15m", 900_000),
        ("30m", 1_800_000),
        ("1h", 3_600_000),
        ("2h", 7_200_000),
        ("4h", 14_400_000),
        ("6h", 21_600_000),
        ("8h", 28_800_000),
        ("12h", 43_200_000),
        ("1d", 86_400_000),
        ("3d", 259_200_000),
        ("1w", 604_800_000),
    ],
)
def test_interval_to_milliseconds(interval_str: str, expected_ms: int):
    assert interval_to_milliseconds(interval_str) == expected_ms


def test_interval_to_milliseconds_invalid():
    with pytest.raises(ValueError):
        interval_to_milliseconds("invalid_interval")


def test_detect_kline_gaps_internal():
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
    step = 300_000  # 5m
    # 模拟缺失第 2, 3 根 K 线
    times = [base_ts + i * step for i in range(10) if i not in (2, 3)]
    df = pd.DataFrame({"open_time": times, "close": [100.0] * len(times)})

    gaps = detect_kline_gaps(df, "5m")
    assert len(gaps) == 1
    assert gaps[0] == (base_ts + 2 * step, base_ts + 3 * step)


def test_detect_kline_gaps_with_boundaries():
    base_ts = 1704067200000
    step = 300_000  # 5m
    times = [base_ts + i * step for i in range(10) if i not in (2, 3)]
    df = pd.DataFrame({"open_time": times, "close": [100.0] * len(times)})

    # 指定超出数据范围的全局起点和终点
    start_bound = base_ts - 2 * step
    end_bound = base_ts + 11 * step
    gaps = detect_kline_gaps(df, "5m", start_time=start_bound, end_time=end_bound)

    assert len(gaps) == 3
    assert gaps[0] == (start_bound, base_ts - step)  # 头部缺失
    assert gaps[1] == (base_ts + 2 * step, base_ts + 3 * step)  # 中间缺失
    assert gaps[2] == (base_ts + 10 * step, end_bound)  # 尾部缺失


def test_detect_kline_gaps_no_gaps():
    base_ts = 1704067200000
    step = 300_000
    times = [base_ts + i * step for i in range(10)]
    df = pd.DataFrame({"open_time": times, "close": [100.0] * len(times)})

    gaps = detect_kline_gaps(df, "5m", start_time=base_ts, end_time=times[-1])
    assert gaps == []


def test_fill_kline_gaps_live_api():
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
    step = 300_000  # 5m
    # 模拟缺失 index 4, 5
    times = [base_ts + i * step for i in range(10) if i not in (4, 5)]
    df = pd.DataFrame(
        {
            "open_time": times,
            "open": [100.0] * len(times),
            "high": [105.0] * len(times),
            "low": [95.0] * len(times),
            "close": [102.0] * len(times),
            "volume": [10.0] * len(times),
            "close_time": [t + step - 1 for t in times],
            "quote_volume": [1000.0] * len(times),
            "count": [50] * len(times),
            "taker_buy_volume": [5.0] * len(times),
            "taker_buy_quote_volume": [500.0] * len(times),
            "ignore": ["0"] * len(times),
        }
    )

    repaired_df, report = fill_kline_gaps(
        df=df,
        symbol="BTCUSDT",
        interval="5m",
        asset_class="um",
        data_type="klines",
    )

    assert report["is_modified"] is True
    assert report["filled_bars_count"] == 2
    assert len(repaired_df) == 10
    # 补齐后应无断层
    assert detect_kline_gaps(repaired_df, "5m") == []


def test_patch_pyarrow_kline_table():
    base_ts = 1704067200000
    step = 300_000
    times = [base_ts + i * step for i in range(6) if i != 3]
    df = pd.DataFrame(
        {
            "open_time": times,
            "open": [100.0] * len(times),
            "high": [105.0] * len(times),
            "low": [95.0] * len(times),
            "close": [102.0] * len(times),
            "volume": [10.0] * len(times),
            "close_time": [t + step - 1 for t in times],
            "quote_volume": [1000.0] * len(times),
            "count": [50] * len(times),
            "taker_buy_volume": [5.0] * len(times),
            "taker_buy_quote_volume": [500.0] * len(times),
            "ignore": ["0"] * len(times),
        }
    )
    table = pa.Table.from_pandas(df, preserve_index=False)

    patched_table = patch_pyarrow_kline_table(
        table=table,
        ticker="BTCUSDT",
        interval="5m",
        asset_class="um",
        data_type="klines",
    )

    df_result = patched_table.to_pandas()
    assert len(df_result) == 6
    assert detect_kline_gaps(df_result, "5m") == []


def test_patch_kline_file(tmp_path):
    base_ts = 1704067200000
    step = 300_000
    times = [base_ts + i * step for i in range(8) if i not in (2, 3)]
    df = pd.DataFrame(
        {
            "open_time": times,
            "open": [100.0] * len(times),
            "high": [105.0] * len(times),
            "low": [95.0] * len(times),
            "close": [102.0] * len(times),
            "volume": [10.0] * len(times),
            "close_time": [t + step - 1 for t in times],
            "quote_volume": [1000.0] * len(times),
            "count": [50] * len(times),
            "taker_buy_volume": [5.0] * len(times),
            "taker_buy_quote_volume": [500.0] * len(times),
            "ignore": ["0"] * len(times),
        }
    )

    file_path = str(tmp_path / "test_klines.parquet")
    df.to_parquet(file_path, index=False)

    report = patch_kline_file(
        file_path=file_path,
        symbol="BTCUSDT",
        interval="5m",
        asset_class="um",
        data_type="klines",
        backup=True,
    )

    assert report["is_modified"] is True
    assert report["filled_bars_count"] == 2
    assert os.path.exists(file_path + ".bak")

    # 验证修复后的文件内容
    df_fixed = pd.read_parquet(file_path)
    assert len(df_fixed) == 8
    assert detect_kline_gaps(df_fixed, "5m") == []
