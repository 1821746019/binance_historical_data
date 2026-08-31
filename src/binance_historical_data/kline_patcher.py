"""
K线数据缺失检测与 Binance REST API 补齐模块

提供通用的 K 线数据断层检测、API 补齐抓取、DataFrame/PyArrow Table 修复以及磁盘文件修复功能。
支持现货(spot)、USDT合约(um)、币本位合约(cm)的各类K线数据(klines, premiumIndexKlines 等)。
"""

import datetime
import logging
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

LOGGER = logging.getLogger(__name__)

# K线标准列名与类型定义
KLINE_COLUMNS: List[str] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

KLINE_DTYPES: Dict[str, str] = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "int64",
    "quote_volume": "float64",
    "count": "int64",
    "taker_buy_volume": "float64",
    "taker_buy_quote_volume": "float64",
    "ignore": "int64",
}

# 常见周期的毫秒数映射
INTERVAL_MS_MAP: Dict[str, int] = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

# 支持 K 线补齐的数据类型
KLINE_DATA_TYPES = {
    "klines",
    "premiumIndexKlines",
    "indexPriceKlines",
    "markPriceKlines",
}


def interval_to_milliseconds(interval: str) -> int:
    """
    将时间频率（如 '1m', '5m', '1h', '1d'）转换为毫秒数
    """
    if interval in INTERVAL_MS_MAP:
        return INTERVAL_MS_MAP[interval]

    match = re.match(r"^(\d+)([smhdw])$", interval.lower())
    if not match:
        raise ValueError(f"无法解析的时间频率格式: {interval}")

    val, unit = int(match.group(1)), match.group(2)
    units = {
        "s": 1_000,
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    return val * units[unit]


def _normalize_timestamp_to_ms(ts: Union[int, float, str, datetime.datetime, datetime.date]) -> int:
    """
    将任意格式的时间输入归一化为 UTC 毫秒时间戳 (int64)
    """
    if isinstance(ts, (int, float)):
        val = int(ts)
        if val < 10**11:  # 秒级
            return val * 1_000
        elif val < 10**14:  # 毫秒级
            return val
        elif val < 10**17:  # 微秒级
            return val // 1_000
        else:  # 纳秒级
            return val // 1_000_000
    elif isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        return int(ts.timestamp() * 1000)
    elif isinstance(ts, datetime.date):
        dt = datetime.datetime(ts.year, ts.month, ts.day, tzinfo=datetime.timezone.utc)
        return int(dt.timestamp() * 1000)
    elif isinstance(ts, str):
        if ts.isdigit():
            return _normalize_timestamp_to_ms(int(ts))
        dt = pd.to_datetime(ts, utc=True)
        return int(dt.timestamp() * 1000)
    else:
        raise TypeError(f"不支持的时间戳格式: {type(ts)} -> {ts}")


def detect_kline_gaps(
    df: pd.DataFrame,
    interval: str,
    timestamp_col: str = "open_time",
    start_time: Optional[Union[int, datetime.datetime, datetime.date, str]] = None,
    end_time: Optional[Union[int, datetime.datetime, datetime.date, str]] = None,
) -> List[Tuple[int, int]]:
    """
    检测 K 线 DataFrame 中缺失的时间戳区间。

    参数:
        df: 包含时间戳列的 DataFrame
        interval: K线频率，例如 '1m', '5m', '1h'
        timestamp_col: 时间戳列名，默认为 'open_time'
        start_time: 期望的起点时间（可选），若数据第一根K线晚于此时间，将记录头部缺失
        end_time: 期望的终点时间（可选），若数据最后一根K线早于此时间，将记录尾部缺失

    返回:
        gaps: 缺失区间列表，每个元素为 (gap_start_ms, gap_end_ms)（闭区间）
    """
    step_ms = interval_to_milliseconds(interval)
    gaps: List[Tuple[int, int]] = []

    start_ms = _normalize_timestamp_to_ms(start_time) if start_time is not None else None
    end_ms = _normalize_timestamp_to_ms(end_time) if end_time is not None else None

    if df.empty or timestamp_col not in df.columns:
        if start_ms is not None and end_ms is not None and start_ms <= end_ms:
            return [(start_ms, end_ms)]
        return gaps

    ts_series = df[timestamp_col]
    if not pd.api.types.is_integer_dtype(ts_series):
        ts_values = ts_series.apply(_normalize_timestamp_to_ms).astype("int64")
    else:
        ts_values = ts_series.astype("int64")

    sorted_ts = ts_values.drop_duplicates().sort_values().to_numpy()
    if len(sorted_ts) == 0:
        if start_ms is not None and end_ms is not None and start_ms <= end_ms:
            return [(start_ms, end_ms)]
        return gaps

    # 1. 检查头部缺失
    first_ts = int(sorted_ts[0])
    if start_ms is not None and start_ms < first_ts:
        gaps.append((start_ms, first_ts - step_ms))

    # 2. 检查中间断层
    diffs = sorted_ts[1:] - sorted_ts[:-1]
    gap_indices = (diffs > step_ms).nonzero()[0]
    for idx in gap_indices:
        gap_start = int(sorted_ts[idx] + step_ms)
        gap_end = int(sorted_ts[idx + 1] - step_ms)
        if gap_start <= gap_end:
            gaps.append((gap_start, gap_end))

    # 3. 检查尾部缺失
    last_ts = int(sorted_ts[-1])
    if end_ms is not None and last_ts < end_ms:
        gaps.append((last_ts + step_ms, end_ms))

    return gaps


def _get_api_endpoint(
    asset_class: Literal["spot", "um", "cm"] = "um",
    data_type: str = "klines",
) -> Tuple[str, int]:
    """
    根据资产类别和数据类型获取对应的 Binance REST API 端点及单次请求 limit 上限
    """
    if asset_class == "um":
        base = "https://fapi.binance.com/fapi/v1"
        limit = 1500
        if data_type == "klines":
            return f"{base}/klines", limit
        elif data_type == "premiumIndexKlines":
            return f"{base}/premiumIndexKlines", limit
        elif data_type == "indexPriceKlines":
            return f"{base}/indexPriceKlines", limit
        elif data_type == "markPriceKlines":
            return f"{base}/markPriceKlines", limit
        else:
            return f"{base}/{data_type}", limit
    elif asset_class == "spot":
        base = "https://api.binance.com/api/v3"
        limit = 1000
        return f"{base}/{data_type}", limit
    elif asset_class == "cm":
        base = "https://dapi.binance.com/dapi/v1"
        limit = 1500
        return f"{base}/{data_type}", limit
    else:
        raise ValueError(f"不支持的 asset_class: {asset_class}")


def fetch_klines_range(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    asset_class: Literal["spot", "um", "cm"] = "um",
    data_type: str = "klines",
    session: Optional[requests.Session] = None,
    max_retries: int = 4,
    request_interval: float = 0.05,
) -> pd.DataFrame:
    """
    从 Binance REST API 分页拉取指定时间范围 [start_time_ms, end_time_ms] 的 K 线数据。
    """
    if start_time_ms > end_time_ms:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    endpoint, limit = _get_api_endpoint(asset_class, data_type)
    step_ms = interval_to_milliseconds(interval)
    http = session or requests.Session()

    all_rows: List[List[Any]] = []
    current_start = start_time_ms

    while current_start <= end_time_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time_ms,
            "limit": limit,
        }

        success = False
        batch_data = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = http.get(endpoint, params=params, timeout=15)
                if resp.status_code == 200:
                    batch_data = resp.json()
                    success = True
                    break
                elif resp.status_code in (429, 418):
                    retry_after = int(resp.headers.get("Retry-After", 2 * attempt))
                    LOGGER.warning("触发 Binance 限频 (%d)，休眠 %d 秒后重试...", resp.status_code, retry_after)
                    time.sleep(retry_after)
                else:
                    LOGGER.warning("请求 API 出错 %s, 状态码: %d, 响应: %s", endpoint, resp.status_code, resp.text)
                    time.sleep(1.0 * attempt)
            except Exception as e:
                LOGGER.warning("请求 API 异常: %s (重试 %d/%d)", e, attempt, max_retries)
                time.sleep(1.0 * attempt)

        if not success or not batch_data:
            break

        all_rows.extend(batch_data)

        last_open_time = int(batch_data[-1][0])
        next_start = last_open_time + step_ms

        if len(batch_data) < limit or next_start > end_time_ms or next_start <= current_start:
            break

        current_start = next_start
        if request_interval > 0:
            time.sleep(request_interval)

    if not all_rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    cols_to_use = KLINE_COLUMNS[: len(all_rows[0])]
    df_fetched = pd.DataFrame(all_rows, columns=cols_to_use)

    for col in cols_to_use:
        if col in KLINE_DTYPES:
            target_type = KLINE_DTYPES[col]
            if target_type == "int64":
                df_fetched[col] = pd.to_numeric(df_fetched[col], errors="coerce").fillna(0).astype("int64")
            elif target_type == "float64":
                df_fetched[col] = pd.to_numeric(df_fetched[col], errors="coerce").astype("float64")
            elif target_type == "str":
                df_fetched[col] = df_fetched[col].astype(str)

    df_fetched = df_fetched[
        (df_fetched["open_time"] >= start_time_ms) & (df_fetched["open_time"] <= end_time_ms)
    ].drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    return df_fetched


def fill_kline_gaps(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    asset_class: Literal["spot", "um", "cm"] = "um",
    data_type: str = "klines",
    start_time: Optional[Union[int, datetime.datetime, datetime.date, str]] = None,
    end_time: Optional[Union[int, datetime.datetime, datetime.date, str]] = None,
    timestamp_col: str = "open_time",
    session: Optional[requests.Session] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    通用 K 线数据缺失检测与自动补齐主函数。

    参数:
        df: 原始 DataFrame
        symbol: 交易对，如 "BTCUSDT"
        interval: 频率，如 "1m", "5m"
        asset_class: 资产类别: "um", "spot", "cm"
        data_type: 数据类型: "klines", "premiumIndexKlines" 等
        start_time: 期望的起始时间（可选）
        end_time: 期望的结束时间（可选）
        timestamp_col: 时间戳列名，默认为 "open_time"
        session: 可选的 requests.Session 实例

    返回:
        (repaired_df, report)
    """
    report: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "gaps_detected": [],
        "filled_bars_count": 0,
        "unfillable_gaps": [],
        "is_modified": False,
    }

    gaps = detect_kline_gaps(
        df=df,
        interval=interval,
        timestamp_col=timestamp_col,
        start_time=start_time,
        end_time=end_time,
    )
    report["gaps_detected"] = gaps

    if not gaps:
        return df, report

    step_ms = interval_to_milliseconds(interval)
    fetched_dfs: List[pd.DataFrame] = []

    for gap_start, gap_end in gaps:
        expected_bars = (gap_end - gap_start) // step_ms + 1
        LOGGER.info(
            "检测到 K 线断层 [%s %s]: %s -> %s (预计缺失 %d 根 bar)，正在调用 API 补齐...",
            symbol,
            interval,
            datetime.datetime.fromtimestamp(gap_start / 1000, tz=datetime.timezone.utc),
            datetime.datetime.fromtimestamp(gap_end / 1000, tz=datetime.timezone.utc),
            expected_bars,
        )

        df_gap = fetch_klines_range(
            symbol=symbol,
            interval=interval,
            start_time_ms=gap_start,
            end_time_ms=gap_end,
            asset_class=asset_class,
            data_type=data_type,
            session=session,
        )

        if not df_gap.empty:
            fetched_dfs.append(df_gap)
            report["filled_bars_count"] += len(df_gap)
            if len(df_gap) < expected_bars:
                report["unfillable_gaps"].append((gap_start, gap_end, expected_bars - len(df_gap)))
        else:
            report["unfillable_gaps"].append((gap_start, gap_end, expected_bars))

    if not fetched_dfs:
        return df, report

    df_orig = df.copy()
    if timestamp_col in df_orig.columns and not pd.api.types.is_integer_dtype(df_orig[timestamp_col]):
        df_orig[timestamp_col] = df_orig[timestamp_col].apply(_normalize_timestamp_to_ms).astype("int64")

    aligned_fetched_dfs = []
    for fdf in fetched_dfs:
        aligned_df = pd.DataFrame(index=fdf.index)
        for col in df_orig.columns:
            if col in fdf.columns:
                try:
                    if pd.api.types.is_integer_dtype(df_orig[col]):
                        aligned_df[col] = pd.to_numeric(fdf[col], errors="coerce").fillna(0).astype(df_orig[col].dtype)
                    elif pd.api.types.is_float_dtype(df_orig[col]):
                        aligned_df[col] = pd.to_numeric(fdf[col], errors="coerce").astype(df_orig[col].dtype)
                    elif pd.api.types.is_string_dtype(df_orig[col]) or df_orig[col].dtype == object:
                        aligned_df[col] = fdf[col].astype(str)
                    else:
                        aligned_df[col] = fdf[col].astype(df_orig[col].dtype)
                except Exception:
                    aligned_df[col] = fdf[col]
            else:
                aligned_df[col] = None
        aligned_fetched_dfs.append(aligned_df)

    combined_df = pd.concat([df_orig] + aligned_fetched_dfs, ignore_index=True)
    combined_df = (
        combined_df.drop_duplicates(subset=[timestamp_col], keep="first")
        .sort_values(by=timestamp_col)
        .reset_index(drop=True)
    )

    for col in df_orig.columns:
        try:
            if pd.api.types.is_integer_dtype(df_orig[col]) or pd.api.types.is_float_dtype(df_orig[col]):
                combined_df[col] = combined_df[col].astype(df_orig[col].dtype)
            elif pd.api.types.is_string_dtype(df_orig[col]) or df_orig[col].dtype == object:
                combined_df[col] = combined_df[col].astype(str)
        except Exception:
            pass

    report["is_modified"] = True
    return combined_df, report


def patch_pyarrow_kline_table(
    table: pa.Table,
    ticker: str,
    interval: str,
    asset_class: Literal["spot", "um", "cm"] = "um",
    data_type: str = "klines",
    start_time: Optional[Union[int, datetime.datetime, datetime.date, str]] = None,
    end_time: Optional[Union[int, datetime.datetime, datetime.date, str]] = None,
) -> pa.Table:
    """
    针对 PyArrow Table 对象就地执行 K 线缺失检测与 API 补齐，返回补齐后的 PyArrow Table。
    """
    if data_type not in KLINE_DATA_TYPES or not interval:
        return table

    df = table.to_pandas()
    if df.empty or "open_time" not in df.columns:
        return table

    repaired_df, report = fill_kline_gaps(
        df=df,
        symbol=ticker,
        interval=interval,
        asset_class=asset_class,
        data_type=data_type,
        start_time=start_time,
        end_time=end_time,
        timestamp_col="open_time",
    )

    if report["is_modified"]:
        LOGGER.info(
            "成功对 %s (%s, %s) 执行补齐: 补充了 %d 根 K 线",
            ticker,
            data_type,
            interval,
            report["filled_bars_count"],
        )
        return pa.Table.from_pandas(repaired_df, schema=table.schema, preserve_index=False)

    return table


def patch_kline_file(
    file_path: str,
    symbol: str,
    interval: str,
    asset_class: Literal["spot", "um", "cm"] = "um",
    data_type: str = "klines",
    start_time: Optional[Union[int, datetime.datetime, datetime.date, str]] = None,
    end_time: Optional[Union[int, datetime.datetime, datetime.date, str]] = None,
    backup: bool = False,
) -> Dict[str, Any]:
    """
    对磁盘上的单个 Parquet / CSV K 线文件进行检查并补齐修复。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    is_parquet = file_path.endswith(".parquet")
    orig_schema = None
    if is_parquet:
        orig_schema = pq.read_schema(file_path)
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    repaired_df, report = fill_kline_gaps(
        df=df,
        symbol=symbol,
        interval=interval,
        asset_class=asset_class,
        data_type=data_type,
        start_time=start_time,
        end_time=end_time,
    )

    if report["is_modified"]:
        if backup:
            bak_path = file_path + ".bak"
            if not os.path.exists(bak_path):
                os.rename(file_path, bak_path)

        if is_parquet:
            table = pa.Table.from_pandas(repaired_df, schema=orig_schema, preserve_index=False)
            pq.write_table(table, file_path, compression="snappy")
        else:
            repaired_df.to_csv(file_path, index=False)

        LOGGER.info("已回写修复后的数据至文件: %s", file_path)

    return report
