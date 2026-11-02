"""官方来源解析、缓存生命周期、联网限制测试；不请求真实网络。"""

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from projects.enterprise_knowledge_agent.app import holiday_calendar as module
from projects.enterprise_knowledge_agent.app.holiday_calendar import (
    CACHE_TTL_SECONDS,
    SOURCE_CATALOG,
    CalendarUnavailable,
    OfficialCalendarProvider,
    check_url,
    download_official,
    parse_sources,
)
from projects.enterprise_knowledge_agent.app.overtime import calculate_overtime_pay
from projects.enterprise_knowledge_agent.tests.calendar_fixtures import (
    fetch_fixture,
    fixture_documents,
)


NOW = datetime(2026, 11, 2, 12, tzinfo=UTC)


@pytest.mark.parametrize("year,breaks,makeups", [(2025, 28, 5), (2026, 33, 6)])
def test_parse_two_official_years(year, breaks, makeups):
    result = parse_sources(year, fixture_documents(year))
    assert len(result["statutory"]) == 13
    assert len(result["breaks"]) == breaks
    assert len(result["workdays"]) == makeups
    assert set(result["statutory"]) <= set(result["breaks"])
    assert not set(result["breaks"]) & set(result["workdays"])


def test_2025_midautumn_is_not_2026_midautumn():
    result = parse_sources(2025, fixture_documents(2025))
    assert "2025-10-06" in result["statutory"]
    assert "2025-10-04" not in result["statutory"]


def test_network_then_cache_survives_new_provider_without_fetch(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return fetch_fixture(url)

    provider = OfficialCalendarProvider(tmp_path, fetcher=fetch, clock=lambda: NOW)
    first = provider.get(2026)
    assert first["data_origin"] == "network" and first["cache_saved"]
    assert len(calls) == 2
    second = OfficialCalendarProvider(
        tmp_path, fetcher=lambda url: pytest.fail("有效缓存不应联网"), clock=lambda: NOW
    ).get(2026)
    assert second["data_origin"] == "cache" and second["stale"] is False
    assert second["fetched_at"] == first["fetched_at"]


def test_shared_2025_source_is_downloaded_once(tmp_path):
    calls = []
    provider = OfficialCalendarProvider(
        tmp_path,
        fetcher=lambda url: calls.append(url) or fetch_fixture(url),
        clock=lambda: NOW,
    )
    provider.get(2025)
    assert len(calls) == 1


def test_expired_cache_refreshes_with_new_timestamp(tmp_path):
    OfficialCalendarProvider(tmp_path, fetcher=fetch_fixture, clock=lambda: NOW).get(
        2026
    )
    later = NOW + timedelta(seconds=CACHE_TTL_SECONDS)
    result = OfficialCalendarProvider(
        tmp_path, fetcher=fetch_fixture, clock=lambda: later
    ).get(2026)
    assert result["data_origin"] == "network"
    assert result["fetched_at"] == later.isoformat()


@pytest.mark.parametrize("expired,force", [(True, False), (False, True)])
def test_failed_refresh_never_returns_old_cache(tmp_path, expired, force):
    OfficialCalendarProvider(tmp_path, fetcher=fetch_fixture, clock=lambda: NOW).get(
        2026
    )

    def offline(url):
        raise httpx.ReadTimeout("测试断网")

    clock = NOW + timedelta(days=2) if expired else NOW
    provider = OfficialCalendarProvider(tmp_path, fetcher=offline, clock=lambda: clock)
    with pytest.raises(CalendarUnavailable, match="未使用过期"):
        provider.get(2026, refresh=force)
    with pytest.raises(CalendarUnavailable):
        provider.get(2026, refresh=True)


@pytest.mark.parametrize(
    "damage", ["invalid_json", "wrong_year", "wrong_url", "future_time", "bad_body"]
)
def test_corrupt_or_forged_cache_is_not_used(tmp_path, damage):
    provider = OfficialCalendarProvider(
        tmp_path, fetcher=fetch_fixture, clock=lambda: NOW
    )
    provider.get(2026)
    path = tmp_path / "2026.json"
    item = json.loads(path.read_text(encoding="utf-8"))
    if damage == "wrong_year":
        item["year"] = 2025
    elif damage == "wrong_url":
        item["documents"] = {"https://evil.example/": "假的日历"}
    elif damage == "future_time":
        item["fetched_at"] = (NOW + timedelta(days=2)).isoformat()
    elif damage == "bad_body":
        item["documents"][SOURCE_CATALOG[2026]["notice"]] = "登录后查看"
    path.write_text(
        "{" if damage == "invalid_json" else json.dumps(item), encoding="utf-8"
    )
    assert provider.get(2026)["data_origin"] == "network"


@pytest.mark.parametrize(
    "change", ["wrong_title", "missing_holiday", "wrong_count", "wrong_statutory"]
)
def test_changed_official_page_requires_reverification(change):
    docs = fixture_documents(2026)
    notice, policy = SOURCE_CATALOG[2026]["notice"], SOURCE_CATALOG[2026]["policy"]
    if change == "wrong_title":
        docs[notice] = docs[notice].replace("关于2026年", "关于2027年")
    elif change == "missing_holiday":
        docs[notice] = docs[notice].replace("七、国庆节：", "国庆节：")
    elif change == "wrong_count":
        docs[notice] = docs[notice].replace("共9天", "共8天")
    else:
        docs[policy] = docs[policy].replace("春节2月16日-19日", "春节2月16日-18日")
    with pytest.raises(CalendarUnavailable):
        parse_sources(2026, docs)


def test_unknown_year_not_assumed_unpublished_and_does_not_fetch(tmp_path):
    provider = OfficialCalendarProvider(
        tmp_path, fetcher=lambda url: pytest.fail("不应猜URL")
    )
    with pytest.raises(CalendarUnavailable, match="不代表官方未公布"):
        provider.get(2027)


@pytest.mark.parametrize(
    "url",
    [
        "http://www.gov.cn/path",
        "https://www.gov.cn.evil.example/",
        "https://127.0.0.1/",
        "https://www.gov.cn:8080/",
        "https://user:pass@www.gov.cn/",
        "file:///etc/passwd",
    ],
)
def test_network_rejects_untrusted_schemes_hosts_ports_and_credentials(url):
    with pytest.raises(CalendarUnavailable):
        check_url(url)


def mock_download(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.delenv("HOLIDAY_HTTP_PROXY", raising=False)
    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(handler), trust_env=False
        ),
    )


def test_redirect_cannot_reach_private_address(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://127.0.0.1/private"})

    mock_download(monkeypatch, respond)
    with pytest.raises(CalendarUnavailable):
        download_official(SOURCE_CATALOG[2026]["notice"])
    assert len(requests) == 1


@pytest.mark.parametrize(
    "headers,body",
    [
        ({"content-type": "application/octet-stream"}, b"not html"),
        ({"content-type": "text/html"}, b"x" * (module.MAX_BYTES + 1)),
    ], ids=["wrong-content-type", "oversized-body"],
)
def test_download_rejects_wrong_type_or_excess_size(monkeypatch, headers, body):
    mock_download(
        monkeypatch, lambda request: httpx.Response(200, headers=headers, content=body)
    )
    with pytest.raises(CalendarUnavailable):
        download_official(SOURCE_CATALOG[2026]["notice"])


def test_invalid_pdf_is_rejected_without_silent_html_fallback(monkeypatch):
    mock_download(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"not a pdf"
        ),
    )
    with pytest.raises(CalendarUnavailable, match="PDF无法"):
        download_official(SOURCE_CATALOG[2026]["policy"])


def test_write_cache_failure_does_not_discard_verified_live_result(tmp_path):
    location = tmp_path / "not-a-directory"
    location.write_text("keep", encoding="utf-8")
    result = OfficialCalendarProvider(
        location, fetcher=fetch_fixture, clock=lambda: NOW
    ).get(2025)
    assert result["data_origin"] == "network" and result["cache_saved"] is False


def test_calendar_network_failure_never_leads_to_money(tmp_path):
    def offline(url):
        raise httpx.ConnectError("测试断网")

    result = calculate_overtime_pay(
        {
            "start_date": "2026-10-01",
            "end_date": "2026-10-05",
            "daily_wage": 300,
            "hours_per_day": 8,
            "wage_basis_confirmed": True,
            "rest_day_compensated": False,
            "work_schedule": "standard_following_calendar",
        },
        provider=OfficialCalendarProvider(tmp_path, fetcher=offline),
    )
    assert result["status"] == "calendar_unavailable"
    assert "total_overtime_pay" not in result
