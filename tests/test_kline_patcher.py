import os
import unittest.mock

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


@pytest.fixture(autouse=True)
def mock_binance_api():
    def _fake_get(url, params=None, timeout=15):
        class FakeResponse:
            def __init__(self, data, status_code=200):
                self._data = data
                self.status_code = status_code
                self.text = str(data)
                self.headers = {}

            def json(self):
                return self._data

        sym = params.get("symbol", "") if params else ""
        if "INVALID" in sym:
            return FakeResponse({"code": -1121, "msg": "Invalid symbol."}, status_code=400)

        start = int(params.get("startTime", 0)) if params else 0
        end = int(params.get("endTime", 0)) if params else 0
        interval_req = params.get("interval", "5m") if params else "5m"
        step = interval_to_milliseconds(interval_req)
        bars = []
        cur = start
        while cur <= end and len(bars) < 1500:
            bars.append(
                [
                    cur,
                    "100.0",
                    "105.0",
                    "95.0",
                    "102.0",
                    "10.0",
                    cur + step - 1,
                    "1000.0",
                    50,
                    "5.0",
                    "500.0",
                    "0",
                ]
            )
            cur += step
        return FakeResponse(bars, status_code=200)

    with (
        unittest.mock.patch("requests.Session.get", side_effect=_fake_get),
        unittest.mock.patch("requests.get", side_effect=_fake_get),
    ):
        yield


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

    # 默认 internal_only=True: 忽略未上市头部与退市尾部，仅捕获序列中间真正断裂的数据
    gaps_internal = detect_kline_gaps(df, "5m", start_time=start_bound, end_time=end_bound)
    assert len(gaps_internal) == 1
    assert gaps_internal[0] == (base_ts + 2 * step, base_ts + 3 * step)

    # 显式 internal_only=False: 检测全量包含头尾边界
    gaps_all = detect_kline_gaps(
        df, "5m", start_time=start_bound, end_time=end_bound, internal_only=False
    )
    assert len(gaps_all) == 3
    assert gaps_all[0] == (start_bound, base_ts - step)  # 头部缺失
    assert gaps_all[1] == (base_ts + 2 * step, base_ts + 3 * step)  # 中间缺失
    assert gaps_all[2] == (base_ts + 10 * step, end_bound)  # 尾部缺失


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


def test_detect_kline_gaps_internal_mode_ignores_boundaries():
    """验证 gap_mode='internal' 默认模式下绝对不向上市前（左侧）或退市后（右侧）外推"""
    base_ts = 1704067200000
    step = 300_000  # 5m
    # 模拟在 base_ts 处上市，且中间缺失 1 根 (第 2 根)
    times = [base_ts, base_ts + step, base_ts + 3 * step, base_ts + 4 * step]
    df = pd.DataFrame({"open_time": times, "close": [100.0] * len(times)})

    # 人为传入整月的月初和月末边界
    start_bound = base_ts - 100 * step
    end_bound = base_ts + 100 * step

    # 默认模式 gap_mode='internal'，必须完全忽略头尾，仅检出内部的一处断层
    gaps_internal = detect_kline_gaps(df, "5m", start_time=start_bound, end_time=end_bound, gap_mode="internal")
    assert len(gaps_internal) == 1
    assert gaps_internal[0] == (base_ts + 2 * step, base_ts + 2 * step)


def test_circuit_breaker_on_delisted_symbol():
    """验证退市或非法标的能迅速熔断并跳过后续无意义的 API 请求"""
    from binance_historical_data.kline_patcher import _UNAVAILABLE_SYMBOLS

    fake_symbol = "TOTALLY_INVALID_COIN_XYZ_USDT"
    assert fake_symbol not in _UNAVAILABLE_SYMBOLS

    # 首次调用触发 400 非法标的错误，应自动记录到熔断黑名单
    df_empty = fetch_klines_range(fake_symbol, "1h", 1704067200000, 1704070800000, max_retries=1)
    assert df_empty.empty
    assert fake_symbol in _UNAVAILABLE_SYMBOLS

    # 再次调用应瞬间返回空 DataFrame，不需要发起网络请求
    df_second = fetch_klines_range(fake_symbol, "1h", 1704067200000, 1704070800000, max_retries=1)
    assert df_second.empty


def test_auto_gap_mode_patches_monthly_slice_boundaries():
    """
    【踩坑实录复现测试】
    复现 2022-02 末（官方归档提前至 02-25 20:00 截断）与
    2022-04 初（官方归档滞后至 04-03 00:00 开始）的边界切片 Bug。
    验证 gap_mode='auto' 传入自然月起止时间时，能够精准检测并补全头尾缺口。
    """
    # 模拟 2022-02 (4h K线, 理论应有 28天 * 6 = 168 根)
    # 归档仅给到了 02-25 20:00 (共 150 根, 缺失 02-26 00:00 至 02-28 20:00 共 18 根)
    base_ts = 1643673600000  # 2022-02-01 00:00:00 UTC
    step = 14_400_000  # 4h
    times_02 = [base_ts + i * step for i in range(150)]
    df_02 = pd.DataFrame(
        {
            "open_time": times_02,
            "open": [10.0] * 150,
            "high": [11.0] * 150,
            "low": [9.0] * 150,
            "close": [10.5] * 150,
            "volume": [100.0] * 150,
            "close_time": [t + step - 1 for t in times_02],
            "quote_volume": [1000.0] * 150,
            "count": [50] * 150,
            "taker_buy_volume": [50.0] * 150,
            "taker_buy_quote_volume": [500.0] * 150,
            "ignore": [0] * 150,
        }
    )

    # 1. 如果使用纯内部模式，因 150 根数据内部完全连续，检出 0 处断层 (导致漏补)
    gaps_pure_internal = detect_kline_gaps(df_02, "4h", gap_mode="internal")
    assert len(gaps_pure_internal) == 0

    # 2. 当传入当月理论起止时间且使用 gap_mode='auto' 时，精准捕获尾部缺失的 18 根 bar
    month_start_ms = base_ts
    month_end_ms = base_ts + 167 * step  # 2022-02-28 20:00:00 UTC
    gaps_auto_02 = detect_kline_gaps(
        df_02, "4h", start_time=month_start_ms, end_time=month_end_ms, gap_mode="auto"
    )
    assert len(gaps_auto_02) == 1
    expected_gap_start = base_ts + 150 * step
    expected_gap_end = month_end_ms
    assert gaps_auto_02[0] == (expected_gap_start, expected_gap_end)
    assert (expected_gap_end - expected_gap_start) // step + 1 == 18

    # 3. 验证 patch_pyarrow_kline_table 在 auto 模式下能自动补齐到满额 168 根
    table_02 = pa.Table.from_pandas(df_02, preserve_index=False)
    patched_table_02 = patch_pyarrow_kline_table(
        table=table_02,
        ticker="WAVESUSDT",
        interval="4h",
        asset_class="um",
        data_type="klines",
        start_time=month_start_ms,
        end_time=month_end_ms,
        gap_mode="auto",
    )
    df_fixed_02 = patched_table_02.to_pandas()
    assert len(df_fixed_02) == 168
    assert df_fixed_02["open_time"].iloc[-1] == month_end_ms

    # 4. 模拟 2022-04 (官方切片从 04-03 00:00 才开始, 缺失月初 12 根)
    april_start_ms = 1648771200000  # 2022-04-01 00:00:00 UTC
    times_04 = [april_start_ms + i * step for i in range(12, 180)]  # 缺少前 12 根
    df_04 = pd.DataFrame(
        {
            "open_time": times_04,
            "open": [50.0] * len(times_04),
            "high": [55.0] * len(times_04),
            "low": [45.0] * len(times_04),
            "close": [52.0] * len(times_04),
            "volume": [200.0] * len(times_04),
            "close_time": [t + step - 1 for t in times_04],
            "quote_volume": [2000.0] * len(times_04),
            "count": [80] * len(times_04),
            "taker_buy_volume": [80.0] * len(times_04),
            "taker_buy_quote_volume": [800.0] * len(times_04),
            "ignore": [0] * len(times_04),
        }
    )
    april_end_ms = april_start_ms + 179 * step
    table_04 = pa.Table.from_pandas(df_04, preserve_index=False)
    patched_table_04 = patch_pyarrow_kline_table(
        table=table_04,
        ticker="WAVESUSDT",
        interval="4h",
        asset_class="um",
        data_type="klines",
        start_time=april_start_ms,
        end_time=april_end_ms,
        gap_mode="auto",
    )
    df_fixed_04 = patched_table_04.to_pandas()
    assert len(df_fixed_04) == 180
    assert df_fixed_04["open_time"].iloc[0] == april_start_ms
