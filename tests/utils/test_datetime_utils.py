"""Tests for the shared datetime helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from src.utils.datetime_utils import parse_iso_utc, to_date_key, to_utc


class TestToUtc:
    def test_naive_gets_utc(self):
        out = to_utc(datetime(2023, 1, 2, 3, 4))
        assert out.tzinfo == timezone.utc
        assert (out.hour, out.minute) == (3, 4)

    def test_aware_converted_to_utc(self):
        eastern = datetime(2023, 1, 2, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        out = to_utc(eastern)
        assert out.tzinfo == timezone.utc
        assert out.hour == 17  # 12:00 EST -> 17:00 UTC


class TestParseIsoUtc:
    def test_trailing_z(self):
        out = parse_iso_utc("2023-01-02T10:00:00Z")
        assert out.tzinfo == timezone.utc
        assert out.hour == 10

    def test_naive_iso_coerced(self):
        assert parse_iso_utc("2023-01-02T10:00:00").tzinfo == timezone.utc


class TestToDateKey:
    def test_datetime(self):
        assert to_date_key(datetime(2023, 1, 2, 9, 30)) == "2023-01-02"

    def test_date(self):
        assert to_date_key(date(2023, 1, 2)) == "2023-01-02"

    def test_str_passthrough(self):
        assert to_date_key("2023-01-02") == "2023-01-02"
