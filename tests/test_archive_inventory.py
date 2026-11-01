import datetime
import io
from unittest.mock import MagicMock, patch

import pytest

from binance_historical_data import BinanceDataDumper
from binance_historical_data.archive_inventory import ArchiveCoverage, list_archive_dates

D = datetime.date


def listing(prefix, keys, tail="<IsTruncated>false</IsTruncated>"):
    entries = "".join(f"<Contents><Key>{prefix}{key}</Key></Contents>" for key in keys)
    return io.BytesIO(f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{entries}{tail}</ListBucketResult>'.encode())


def test_pagination_filters_checksums():
    prefix = "data/futures/um/daily/metrics/TESTUSDT/"
    with patch("urllib.request.urlopen", side_effect=[
        listing(prefix, ["TESTUSDT-metrics-2025-01-01.zip", "TESTUSDT-metrics-2025-01-01.zip.CHECKSUM"],
                "<IsTruncated>true</IsTruncated><NextContinuationToken>next</NextContinuationToken>"),
        listing(prefix, ["TESTUSDT-metrics-2025-01-02.zip"]),
    ]) as request:
        assert list_archive_dates(prefix, "daily") == {D(2025, 1, 1), D(2025, 1, 2)}
    assert "continuation-token=next" in request.call_args.args[0]


def test_monthly_dates_map_to_first_of_month():
    prefix = "data/spot/monthly/klines/BTCUSDT/1h/"
    with patch("urllib.request.urlopen", return_value=listing(prefix, ["BTCUSDT-1h-2025-01.zip"])):
        assert list_archive_dates(prefix, "monthly") == {D(2025, 1, 1)}


def test_empty_listing_is_not_an_error():
    with patch("urllib.request.urlopen", return_value=listing("p/", [])):
        assert list_archive_dates("p/", "monthly") == frozenset()


def test_placeholder_keys_are_ignored():
    prefix = "data/futures/um/daily/metrics/TRBUSDT/"
    keys = ["TRBUSDT-metrics-1993.zip", "TRBUSDT-metrics-2021-12-01.zip"]
    with patch("urllib.request.urlopen", return_value=listing(prefix, keys)):
        assert list_archive_dates(prefix, "daily") == {D(2021, 12, 1)}


def test_incomplete_pagination_raises():
    with patch("urllib.request.urlopen", return_value=listing("p/", [], "<IsTruncated>true</IsTruncated>")):
        with pytest.raises(ValueError, match="continuation token"):
            list_archive_dates("p/", "monthly")


def test_network_failure_raises_instead_of_returning_empty():
    with (
        patch("urllib.request.urlopen", side_effect=OSError("offline")) as request,
        patch("binance_historical_data.archive_inventory.time.sleep"),
    ):
        with pytest.raises(OSError, match="offline"):
            list_archive_dates("p/", "monthly")
    assert request.call_count == 3


def test_unknown_timeperiod_raises():
    with pytest.raises(ValueError, match="timeperiod"):
        list_archive_dates("p/", "weekly")


def test_coverage_combines_monthly_and_daily():
    coverage = ArchiveCoverage(monthly=frozenset({D(2024, 1, 1)}), daily=frozenset({D(2024, 3, 5)}))
    assert coverage.covers(D(2024, 1, 31))
    assert coverage.covers(D(2024, 3, 5))
    assert not coverage.covers(D(2024, 2, 1))
    assert not coverage.covers(D(2024, 3, 4))
    assert coverage.earliest() == D(2024, 1, 1)
    assert ArchiveCoverage(frozenset(), frozenset()).earliest() is None


@pytest.mark.parametrize("data_type, monthly_listed", [("klines", True), ("metrics", False)])
def test_dumper_coverage_lists_vision_prefixes(data_type, monthly_listed):
    dumper = MagicMock(ONLY_DAILY_DATA_TYPES=BinanceDataDumper.ONLY_DAILY_DATA_TYPES, _data_type=data_type)
    dumper._get_path_suffix_to_dir_with_data.side_effect = lambda period, ticker: f"futures\\um\\{period}\\{data_type}\\{ticker}"
    with patch("binance_historical_data.data_dumper.list_archive_dates", return_value=frozenset({D(2024, 1, 1)})) as listed:
        coverage = BinanceDataDumper.get_archive_coverage(dumper, "BTCUSDT")
    prefixes = [call.args for call in listed.call_args_list]
    assert (f"data/futures/um/daily/{data_type}/BTCUSDT/", "daily") in prefixes
    assert ((f"data/futures/um/monthly/{data_type}/BTCUSDT/", "monthly") in prefixes) == monthly_listed
    assert coverage.daily == {D(2024, 1, 1)}
