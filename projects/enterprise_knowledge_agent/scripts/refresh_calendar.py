"""不调用模型，单独验证官方日历的真实联网与缓存。"""

import argparse
import json

from projects.enterprise_knowledge_agent.app.holiday_calendar import (
    CalendarUnavailable,
    default_calendar_provider,
)


def main():
    parser = argparse.ArgumentParser(
        description="获取并校验官方年度日历，不发送个人问题或工资"
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--refresh", action="store_true", help="忽略有效缓存，强制重新联网"
    )
    args = parser.parse_args()
    try:
        result = default_calendar_provider().get(args.year, refresh=args.refresh)
    except CalendarUnavailable as error:
        print(f"日历不可用：{error}")
        return 1
    print(
        json.dumps(
            {
                "year": result["year"],
                "data_origin": result["data_origin"],
                "fetched_at": result["fetched_at"],
                "expires_at": result["expires_at"],
                "stale": result["stale"],
                "statutory_dates": result["statutory"],
                "break_days": len(result["breaks"]),
                "makeup_workdays": result["workdays"],
                "sources": [source["url"] for source in result["sources"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
