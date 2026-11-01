"""Inventory of archives published on data.binance.vision"""
from __future__ import annotations

import datetime
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

_BUCKET_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/"
_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
# 币安归档中混入了占位对象（如 xxx-metrics-1993.zip），只接受合法日期文件名
_ARCHIVE_DATE = {
    "daily": re.compile(r"-\d{4}-\d{2}-\d{2}$"),
    "monthly": re.compile(r"-\d{4}-\d{2}$"),
}


def _parse_archive_date(key: str, timeperiod: str) -> datetime.date | None:
    stem = key[:-4]
    if not _ARCHIVE_DATE[timeperiod].search(stem):
        return None
    try:
        return datetime.date.fromisoformat(
            stem[-10:] if timeperiod == "daily" else stem[-7:] + "-01"
        )
    except ValueError:
        return None


def list_archive_dates(prefix: str, timeperiod: str) -> frozenset[datetime.date]:
    """List archive dates under an S3 prefix; monthly archives map to the 1st of month"""
    if timeperiod not in _ARCHIVE_DATE:
        raise ValueError(f"Unknown timeperiod: {timeperiod}")
    dates: set[datetime.date] = set()
    token = ""
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = _BUCKET_URL + "?" + urllib.parse.urlencode(params)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    root = ET.fromstring(response.read())
                break
            except (OSError, ET.ParseError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        for entry in root.findall("s3:Contents/s3:Key", _NAMESPACE):
            key = entry.text or ""
            if not key.startswith(prefix) or not key.endswith(".zip"):
                continue
            parsed = _parse_archive_date(key, timeperiod)
            if parsed is not None:
                dates.add(parsed)
        if root.findtext("s3:IsTruncated", namespaces=_NAMESPACE) != "true":
            return frozenset(dates)
        next_token = root.findtext("s3:NextContinuationToken", namespaces=_NAMESPACE)
        if not next_token or next_token == token:
            raise ValueError("Truncated Vision inventory has no advancing continuation token")
        token = next_token


@dataclass(frozen=True)
class ArchiveCoverage:
    """Periods for which a ticker has published archives"""

    monthly: frozenset[datetime.date]
    daily: frozenset[datetime.date]

    def covers(self, day: datetime.date) -> bool:
        return day in self.daily or day.replace(day=1) in self.monthly

    def earliest(self) -> datetime.date | None:
        return min(self.monthly | self.daily, default=None)
