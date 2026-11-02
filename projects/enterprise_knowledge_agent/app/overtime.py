"""官方日历联网/缓存查询与受限加班费计算；日期类型和倍数由程序确定。"""

from datetime import date, timedelta  # 标准日期运算处理月末。
from decimal import Decimal, ROUND_HALF_UP  # 金额不用二进制浮点数。
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .holiday_calendar import CalendarUnavailable, default_calendar_provider


LAW_URL = "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/bgt/art/2023/art_d9aa750028b14b99a776cb93726a360d.html"


class CalendarArguments(BaseModel):
    """缺失年份追问，不默认当前年份；刷新开关必须是真布尔值。"""

    model_config = ConfigDict(extra="forbid")  # 不允许传入自定义倍率或日期标签。
    start_date: str | None = Field(
        default=None, description="用户明确的起始日期 YYYY-MM-DD，不能猜年份"
    )
    end_date: str | None = Field(
        default=None, description="用户明确的结束日期 YYYY-MM-DD，包含当天"
    )
    refresh_calendar: StrictBool = Field(
        default=False,
        description="用户明确要求更新官方日历时设true，否则复用24小时内缓存",
    )


class OvertimeArguments(CalendarArguments):
    """按全国调休日历执行的标准工时、每天相同且不超过八小时。"""

    daily_wage: Decimal | None = Field(
        default=None, gt=0, le=1000000, max_digits=10, decimal_places=2
    )
    hours_per_day: Decimal | None = Field(
        default=None, gt=0, le=8, max_digits=4, decimal_places=2
    )
    wage_basis_confirmed: StrictBool | None = Field(
        default=None, description="用户确认日工资为适用的加班费计算基数"
    )
    rest_day_compensated: StrictBool | None = Field(
        default=None, description="休息日是否全部已安排等时补休；各天不同需拆分"
    )
    work_schedule: Literal["standard_following_calendar", "other"] | None = Field(
        default=None,
        description="用户确认标准工时并遵循全国调休日历后选standard_following_calendar",
    )


def _blocked(status: str, questions: list[str]) -> dict:
    """缺参或超出支持范围不给出金额。"""
    return {"status": status, "questions": questions, "matches": []}


def query_holiday_calendar(raw_arguments: dict, *, provider=None) -> dict:
    """工具可以联网获取年度日历；实际企业排班及特殊工时必须另行核实。"""
    args = CalendarArguments.model_validate(raw_arguments)
    try:
        start, end = (
            date.fromisoformat(args.start_date or ""),
            date.fromisoformat(args.end_date or ""),
        )
    except ValueError:
        return _blocked(
            "needs_clarification",
            ["请提供带年份的有效起止日期，例如 2026-10-01 至 2026-10-05。"],
        )
    if start > end:
        return _blocked(
            "needs_clarification", ["结束日期不能早于开始日期，请确认日期范围。"]
        )
    if start.year != end.year:
        return _blocked("unsupported", ["跨年日期请按年份拆分，不能混用不同年度日历。"])
    if (end - start).days >= 31:
        return _blocked("unsupported", ["单次最多查询31天，请拆分日期范围。"])
    try:
        calendar = (provider or default_calendar_provider()).get(
            start.year, refresh=args.refresh_calendar
        )
    except CalendarUnavailable as error:
        return _blocked(
            "calendar_unavailable", [str(error)]
        )  # 网络失败不能回退到硬编码。
    days = []
    for offset in range((end - start).days + 1):
        current = start + timedelta(days=offset)
        iso = current.isoformat()
        if iso in calendar["statutory"]:
            kind = "statutory_holiday"
        elif iso in calendar["workdays"]:
            kind = "workday"  # 补班周末不是休息日。
        elif iso in calendar["breaks"] or current.weekday() >= 5:
            kind = "rest_day"
        else:
            kind = "workday"
        days.append({"date": iso, "day_type": kind})
    return {
        "status": "ok",
        "calendar_year": start.year,
        "calendar": {
            key: calendar[key]
            for key in ("data_origin", "fetched_at", "expires_at", "stale")
        },
        "assumption": "中国大陆全国放假调休安排；企业实际排班与特殊工时需另行核实。",
        "days": days,
        "matches": calendar["sources"],
    }


def _money(value: Decimal) -> str:
    """金额按分四舍五入。"""
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def calculate_overtime_pay(raw_arguments: dict, *, provider=None) -> dict:
    """重新获取有效日历后计算；不信任模型传入的日期类型或倍率。"""
    args = OvertimeArguments.model_validate(raw_arguments)
    calendar = query_holiday_calendar(
        args.model_dump(include={"start_date", "end_date", "refresh_calendar"}),
        provider=provider,
    )
    if calendar["status"] != "ok":
        return calendar
    questions = []
    if args.daily_wage is None:
        questions.append("适用的日工资计算基数是多少元？")
    if args.wage_basis_confirmed is not True:
        questions.append(
            "请确认日工资是适用的加班工资计算基数；基本工资不一定等于计算基数。"
        )
    if args.hours_per_day is None:
        questions.append("每天实际工作多少小时？首版支持每天相同且不超过8小时。")
    if args.work_schedule is None:
        questions.append("是否实行标准工时，且实际休息安排遵循全国放假调休日历？")
    if args.work_schedule == "other":
        return _blocked(
            "unsupported",
            ["特殊工时或自定义排班不适用当前工具，请先核实规则和实际排班。"],
        )
    if (
        any(day["day_type"] == "rest_day" for day in calendar["days"])
        and args.rest_day_compensated is None
    ):
        questions.append("休息日加班是否全部安排等时补休？如果各天不同，请拆开计算。")
    if any(day["day_type"] == "workday" for day in calendar["days"]):
        return _blocked(
            "unsupported",
            [
                "区间含工作日；当前工具不把正常上班当加班，请拆分并核实正常工时外的加班时长。"
            ],
        )
    if questions:
        return _blocked("needs_clarification", questions)
    rows, total = [], Decimal("0")
    for day in calendar["days"]:
        multiplier = (
            Decimal("3")
            if day["day_type"] == "statutory_holiday"
            else Decimal("0" if args.rest_day_compensated else "2")
        )
        amount = (
            args.daily_wage / Decimal("8") * args.hours_per_day * multiplier
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        total += amount
        rows.append(
            {
                **day,
                "hours": str(args.hours_per_day),
                "multiplier": str(multiplier),
                "amount": _money(amount),
            }
        )
    law_path = Path(__file__).resolve().parent.parent / "knowledge" / "labor_law.txt"
    law_text = next(
        line.strip()
        for line in law_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("第四十四条")
    )
    return {
        "status": "ok",
        "kind": "overtime_calculation",
        "daily_wage": str(args.daily_wage),
        "hours_per_day": str(args.hours_per_day),
        "rest_day_compensated": args.rest_day_compensated,
        "days": rows,
        "total_overtime_pay": _money(total),
        "currency": "CNY",
        "calendar": calendar["calendar"],
        "scope": "仅按已确认前提计算的加班工资最低标准，逐日四舍五入到分后汇总；不含正常工资，也不是应补发差额或个案法律认定。",
        "matches": [
            {
                "source_id": "劳动法#第四十四条",
                "text": law_text,
                "kind": "law",
                "url": LAW_URL,
            },
            *calendar["matches"],
        ],
    }


def format_overtime_report(result: dict) -> str:
    """最终金额、明细及日历新鲜度由服务器生成，不由模型改写。"""
    labels = {"statutory_holiday": "法定节假日", "rest_day": "休息日"}
    freshness = result["calendar"]
    lines = [
        f"按您确认的条件，加班工资最低标准合计为 **{result['total_overtime_pay']} 元**。",
        f"计算前提：标准工时、遵循全国调休日历，日计算基数 {result['daily_wage']} 元，每天 {result['hours_per_day']} 小时。",
        f"休息日已安排等时补休：{'是' if result['rest_day_compensated'] else '否'}。法定节假日不因补休免除加班工资。",
        "",
        "| 日期 | 类型 | 工时 | 倍率 | 加班费（元） |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["days"]:
        lines.append(
            f"| {row['date']} | {labels[row['day_type']]} | {row['hours']} | {row['multiplier']} | {row['amount']} |"
        )
    lines.extend(
        [
            "",
            "公式：日计算基数 ÷ 8 × 当日工作小时数 × 当日倍率。",
            result["scope"],
            f"日历来源：{'本次联网获取' if freshness['data_origin'] == 'network' else '有效期内缓存'}；获取时间：{freshness['fetched_at']}；有效期至：{freshness['expires_at']}。",
            "",
            "核验来源：",
        ]
    )
    for source in result["matches"]:
        lines.append(f"- [{source['source_id']}]({source['url']})")
    return "\n\n".join(lines[:3]) + "\n" + "\n".join(lines[3:])
