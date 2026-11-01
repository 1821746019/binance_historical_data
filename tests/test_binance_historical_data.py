import datetime
from pathlib import Path
from unittest.mock import MagicMock
import zipfile

import pyarrow.parquet as pq
import pytest

from binance_historical_data import BinanceDataDumper
from binance_historical_data.archive_inventory import ArchiveCoverage
from binance_historical_data.data_dumper import _archive_period_bounds

D = datetime.date
UTC = datetime.timezone.utc
NO_COVERAGE = ArchiveCoverage(frozenset(), frozenset())


def months(*pairs):
    return frozenset(D(year, month, 1) for year, month in pairs)


def test_listing_month_skips_head_but_keeps_tail():
    coverage = ArchiveCoverage(months((2024, 11), (2024, 12)), frozenset())
    assert _archive_period_bounds(D(2024, 11, 1), "monthly", "1d", coverage) == (
        None, datetime.datetime(2024, 11, 30, tzinfo=UTC))


def test_mid_life_month_keeps_both_bounds():
    coverage = ArchiveCoverage(months((2022, 3), (2022, 4), (2022, 5)), frozenset())
    assert _archive_period_bounds(D(2022, 4, 1), "monthly", "8h", coverage) == (
        datetime.datetime(2022, 4, 1, tzinfo=UTC), datetime.datetime(2022, 4, 30, 16, tzinfo=UTC))


def test_relisting_gap_skips_both_sides():
    coverage = ArchiveCoverage(months((2022, 7), (2023, 3)), frozenset())
    assert _archive_period_bounds(D(2022, 7, 1), "monthly", "1d", coverage)[1] is None
    assert _archive_period_bounds(D(2023, 3, 1), "monthly", "1d", coverage)[0] is None


def test_monthly_and_daily_archives_border_each_other():
    coverage = ArchiveCoverage(months((2026, 9)), frozenset({D(2026, 10, 1)}))
    assert _archive_period_bounds(D(2026, 9, 1), "monthly", "1d", coverage)[1] == datetime.datetime(2026, 9, 30, tzinfo=UTC)
    assert _archive_period_bounds(D(2026, 10, 1), "daily", "1h", coverage) == (
        datetime.datetime(2026, 10, 1, tzinfo=UTC), None)


def test_non_kline_archive_has_no_bounds():
    assert _archive_period_bounds(D(2025, 1, 1), "daily", "", NO_COVERAGE) == (None, None)


def archive_dumper(tmp_path, file_format, patch_gaps=False):
    dumper = MagicMock()
    dumper.path_dir_where_to_dump = str(tmp_path)
    dumper._base_url = "https://example.invalid"
    dumper._file_format = file_format
    dumper._data_type = "klines" if patch_gaps else "metrics"
    dumper._data_frequency = "1h"
    dumper._asset_class = "um"
    dumper._patch_gaps = patch_gaps
    dumper._get_path_suffix_to_dir_with_data.return_value = ""
    dumper.create_filename.return_value = "archive.zip"

    def download(url, destination):
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("data.csv", "symbol,sum_open_interest\nBTCUSDT,12\n")
        return True

    dumper._download_raw_file.side_effect = download
    return dumper


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("file_format,patch_gaps", [("parquet", False), ("csv", True), ("csv", False)])
def test_archive_interruption_does_not_publish_partial_output(tmp_path, monkeypatch, existing, file_format, patch_gaps):
    dumper = archive_dumper(tmp_path, file_format, patch_gaps)
    destination = tmp_path / f"data.{file_format}"
    if existing:
        destination.write_bytes(b"previous complete file")

    def interrupt_conversion(**kwargs):
        output = kwargs["parquet_file_path" if file_format == "parquet" else "csv_file_path"]
        Path(output).write_bytes(b"partial")
        raise KeyboardInterrupt

    dumper._convert_zip_entry_to_parquet.side_effect = interrupt_conversion
    dumper._convert_zip_entry_to_csv_with_patch.side_effect = interrupt_conversion
    if file_format == "csv" and not patch_gaps:
        monkeypatch.setattr(zipfile.ZipFile, "testzip", lambda archive: None)
        original_read = zipfile.ZipExtFile.read

        def interrupt_read(stream, size=-1):
            if size == 1024 * 1024:
                raise KeyboardInterrupt
            return original_read(stream, size)

        monkeypatch.setattr(zipfile.ZipExtFile, "read", interrupt_read)

    with pytest.raises(KeyboardInterrupt):
        BinanceDataDumper._download_data_for_1_ticker_1_date(
            dumper, "BTCUSDT", datetime.date(2025, 1, 1), "daily", NO_COVERAGE
        )

    if existing:
        assert destination.read_bytes() == b"previous complete file"
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))
    assert not (tmp_path / "archive.zip").exists()


@pytest.mark.parametrize("file_format", ["parquet", "csv"])
@pytest.mark.parametrize("empty", [False, True])
def test_archive_publishes_completed_output(tmp_path, file_format, empty):
    dumper = archive_dumper(tmp_path, file_format)
    dumper._convert_zip_entry_to_parquet.side_effect = BinanceDataDumper._convert_zip_entry_to_parquet
    contents = "" if empty else "symbol,sum_open_interest\nBTCUSDT,12\n"

    def download(url, destination):
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("data.csv", contents)
        return True

    dumper._download_raw_file.side_effect = download
    destination = tmp_path / f"data.{file_format}"
    destination.write_bytes(b"previous complete file")
    date = datetime.date(2025, 1, 1)
    assert BinanceDataDumper._download_data_for_1_ticker_1_date(dumper, "BTCUSDT", date, "daily", NO_COVERAGE) == date
    if file_format == "parquet":
        assert pq.read_table(destination).num_rows == (0 if empty else 1)
    else:
        assert destination.read_text() == contents
    assert not list(tmp_path.glob("*.part"))
    assert not (tmp_path / "archive.zip").exists()


def test_listing_month_archive_is_patched_without_head_bound(tmp_path, monkeypatch):
    dumper = archive_dumper(tmp_path, "parquet", patch_gaps=True)
    dumper._convert_zip_entry_to_parquet.side_effect = BinanceDataDumper._convert_zip_entry_to_parquet

    def download(url, destination):
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("data.csv", "1731628800000,1,1,1,1,1,1731715199999,1,1,1,1,0\n")
        return True

    dumper._download_raw_file.side_effect = download
    calls = []
    monkeypatch.setattr("binance_historical_data.data_dumper.patch_pyarrow_kline_table",
                        lambda **kwargs: calls.append(kwargs) or kwargs["table"])
    coverage = ArchiveCoverage(months((2024, 11), (2024, 12)), frozenset())
    BinanceDataDumper._download_data_for_1_ticker_1_date(dumper, "BTCUSDT", D(2024, 11, 1), "monthly", coverage)
    assert calls[0]["start_time"] is None
    assert calls[0]["end_time"] == datetime.datetime(2024, 11, 30, 23, tzinfo=UTC)


def test_import():
    from binance_historical_data import BinanceDataDumper

def test_main_class_init():
    from binance_historical_data import BinanceDataDumper

    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=".",
        data_type="klines",  # aggTrades, klines, trades
        data_frequency="1m",  # argument for data_type="klines"
    )

    ## usdm futures
    data_dumper = BinanceDataDumper(
        asset_class="um",  # spot, um, cm
        path_dir_where_to_dump="./BianceFutures",
        data_type="klines",  # aggTrades, klines, trades
        data_frequency="1m",  # argument for data_type="klines"
    )

    try:
        data_dumper.dump_data(
            tickers='BTCUSDT',
            date_start=None,
            date_end=None,
            is_to_update_existing=False,
            tickers_to_exclude=["UST"],
        )
    except Exception as e:
        # 网络环境受限或无法直连币安线上接口时允许跳过网络调用
        pass