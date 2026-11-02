"""固定官方日历与假模型回归测试：不下载模型，不调用付费 API。"""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from projects.enterprise_knowledge_agent.app.graph import build_agent_graph
from projects.enterprise_knowledge_agent.app.overtime import (
    OvertimeArguments,
    calculate_overtime_pay,
    query_holiday_calendar,
)
from projects.enterprise_knowledge_agent.app.tools import ToolRegistry
from projects.enterprise_knowledge_agent.app.holiday_calendar import OfficialCalendarProvider
from projects.enterprise_knowledge_agent.tests.calendar_fixtures import fetch_fixture


@pytest.fixture(autouse=True)
def isolated_official_calendar(monkeypatch, tmp_path):
    provider = OfficialCalendarProvider(tmp_path / "calendar", fetcher=fetch_fixture)
    monkeypatch.setattr("projects.enterprise_knowledge_agent.app.overtime.default_calendar_provider", lambda: provider)
    return provider


def confirmed_arguments():
    """这是用户已明确确认的测试前提，不是生产代码默认值。"""
    return {
        "start_date": "2026-10-01",
        "end_date": "2026-10-05",
        "daily_wage": "300",
        "hours_per_day": "8",
        "wage_basis_confirmed": True,
        "rest_day_compensated": False,
        "work_schedule": "standard_following_calendar",
    }


def test_national_day_uses_three_statutory_and_two_rest_days():
    result = calculate_overtime_pay(confirmed_arguments())
    assert result["total_overtime_pay"] == "3900.00"
    assert [day["amount"] for day in result["days"]] == ["900.00"] * 3 + ["600.00"] * 2
    assert [day["day_type"] for day in result["days"]] == ["statutory_holiday"] * 3 + [
        "rest_day"
    ] * 2


def test_rest_compensation_does_not_cancel_statutory_pay():
    result = calculate_overtime_pay(
        {**confirmed_arguments(), "rest_day_compensated": True}
    )
    assert result["total_overtime_pay"] == "2700.00"


def test_four_hours_and_decimal_wages():
    result = calculate_overtime_pay(
        {**confirmed_arguments(), "hours_per_day": "4", "daily_wage": "300.01"}
    )
    assert (
        result["total_overtime_pay"] == "1950.08"
    )  # 逐日按分舍入后汇总，保证明细可相加。


@pytest.mark.parametrize(
    "missing",
    [
        "start_date",
        "end_date",
        "daily_wage",
        "hours_per_day",
        "wage_basis_confirmed",
        "rest_day_compensated",
        "work_schedule",
    ],
)
def test_missing_confirmation_or_input_does_not_produce_total(missing):
    args = confirmed_arguments()
    args.pop(missing)
    result = calculate_overtime_pay(args)
    assert result["status"] == "needs_clarification"
    assert "total_overtime_pay" not in result


@pytest.mark.parametrize(
    "start,end,status",
    [
        ("10-01", "10-05", "needs_clarification"),
        ("2024-10-01", "2024-10-05", "calendar_unavailable"),
        ("2027-10-01", "2027-10-05", "calendar_unavailable"),
        ("2026-02-30", "2026-03-01", "needs_clarification"),
        ("2026-10-05", "2026-10-01", "needs_clarification"),
        ("2026-01-01", "2026-03-01", "unsupported"),
    ],
)
def test_calendar_rejects_missing_year_unknown_year_invalid_or_large_ranges(
    start, end, status
):
    assert (
        query_holiday_calendar({"start_date": start, "end_date": end})["status"]
        == status
    )


@pytest.mark.parametrize(
    "day",
    [
        "2026-01-04",
        "2026-02-14",
        "2026-02-28",
        "2026-05-09",
        "2026-09-20",
        "2026-10-10",
    ],
)
def test_makeup_weekends_are_workdays(day):
    result = query_holiday_calendar({"start_date": day, "end_date": day})
    assert result["days"][0]["day_type"] == "workday"


def test_thirteen_statutory_dates_and_extended_spring_festival(isolated_official_calendar):
    assert len(isolated_official_calendar.get(2026)["statutory"]) == 13
    result = query_holiday_calendar(
        {"start_date": "2026-02-15", "end_date": "2026-02-23"}
    )
    assert sum(day["day_type"] == "statutory_holiday" for day in result["days"]) == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"daily_wage": "-1"},
        {"hours_per_day": "12"},
        {"hours_per_day": "0"},
        {"wage_basis_confirmed": "true"},
        {"multiplier": 100},
        {"daily_wage": "NaN"},
    ],
)
def test_caller_cannot_supply_invalid_values_or_choose_multiplier(changes):
    with pytest.raises(ValidationError):
        OvertimeArguments.model_validate({**confirmed_arguments(), **changes})


def test_special_schedule_and_ordinary_work_are_not_miscalculated():
    assert (
        calculate_overtime_pay({**confirmed_arguments(), "work_schedule": "other"})[
            "status"
        ]
        == "unsupported"
    )
    result = calculate_overtime_pay(
        {**confirmed_arguments(), "start_date": "2026-10-10", "end_date": "2026-10-10"}
    )
    assert result["status"] == "unsupported"


class EmptyKnowledgeBase:
    def search(self, *args, **kwargs):
        return []


@pytest.mark.parametrize("confirmed", [True, False])
def test_graph_calls_calculation_and_uses_server_generated_result(confirmed):
    """工具与 call_id 真正走通；模型伪造的金额不得作为最终结果。"""
    args = confirmed_arguments()
    if not confirmed:
        args.pop("hours_per_day")
    responses = [
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="calculate_overtime_pay",
                    arguments=json.dumps(args),
                    call_id="calc-1",
                )
            ],
            output_text="",
        ),
        SimpleNamespace(
            output=[],
            output_text=json.dumps(
                {"answer": "伪造金额999999元", "refused": False, "citations": []}
            ),
        ),
    ]
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return responses[len(requests) - 1]

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    graph = build_agent_graph(client, "fake", ToolRegistry(EmptyKnowledgeBase()))
    state = graph.invoke(
        {
            "conversation_input": [
                {"role": "user", "content": "计算已确认条件的加班费"}
            ],
            "tool_calls": [],
            "tool_results": [],
            "retrieved_matches": [],
            "final_answer": None,
            "round_number": 0,
            "max_rounds": 5,
        }
    )
    answer = state["final_answer"]
    assert "999999" not in answer["answer"]
    assert answer["refused"] is (not confirmed)
    assert (
        "3900.00" in answer["answer"] if confirmed else "多少小时" in answer["answer"]
    )
    assert requests[1]["input"][-1]["call_id"] == "calc-1"
    assert len(answer["citations"]) == (3 if confirmed else 0)


def test_registry_exposes_new_tools_and_returns_readable_invalid_input():
    registry = ToolRegistry(EmptyKnowledgeBase())
    assert {tool["name"] for tool in registry.build_definitions()} == {
        "search_knowledge_base",
        "query_holiday_calendar",
        "calculate_overtime_pay",
    }
    result = registry.execute(
        {"name": "calculate_overtime_pay", "arguments": {"daily_wage": -1}}
    )
    assert result["status"] == "needs_clarification"
