"""官方日历联网适配器：固定可信来源、确定性解析、校验后缓存、失败不猜测。"""

import json  # 缓存公开网页，不保存用户问题和工资。
from html import escape  # 将PDF文字变成纯文本载体。
from io import BytesIO  # 有大小限制的PDF仅在内存中解析。
import os  # 原子替换缓存文件，并读取可选代理。
import re  # 只解析已适配的官方日期格式，不调用模型解读网页。
import tempfile  # 并发写入时先完成临时文件，再原子替换。
import time  # 为流式下载设置总耗时上限。
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache  # 服务进程复用一个带锁的提供器。
from html.parser import HTMLParser  # 使用标准库提取正文与表格，无需增加依赖。
from pathlib import Path
from threading import RLock
from urllib.parse import urljoin, urlsplit

import httpx  # 与本地 API 客户端不同，这个客户端专门访问政府网站。

from .config import load_env


SOURCE_CATALOG = {  # 年份不能由模型映射成任意 URL，新增来源需审核并补解析测试。
    2025: {
        "notice": "https://public.zhengzhou.gov.cn/D3603Y/9134461.jhtml",
        "policy": "https://public.zhengzhou.gov.cn/D3603Y/9134461.jhtml",
        "adapter": "table",
    },
    2026: {
        "notice": "https://www.gov.cn/gongbao/2025/issue_12406/202511/content_7048922.html",
        "policy": "https://rsj.sh.gov.cn/dzb/rsj/2026-01-15/pdf/02.pdf",
        "adapter": "paragraphs_pdf",
    },
}
ALLOWED_HOSTS = frozenset(
    urlsplit(url).hostname
    for spec in SOURCE_CATALOG.values()
    for key, url in spec.items()
    if key != "adapter"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_TTL_SECONDS = 24 * 60 * 60  # 超过一天必须重新联网，过期缓存不参与计算。
MAX_BYTES = 2_000_000
TRANSLATION = str.maketrans("節慶調曆國勞週", "节庆调历国劳周")


class CalendarUnavailable(ValueError):
    """可信日历不足时阻止算钱；不能把网络失败当作普通工作日。"""


class OfficialHTML(HTMLParser):
    """同时提取纯文本与表格单元格，跳过脚本和样式。"""

    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.parts, self.rows = [], []
        self.row, self.cell = None, None
        self.hidden = 0
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag == "tr":
            self.row = []
        if tag in {"td", "th"}:
            self.cell = []
        if tag in {"p", "div", "br", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)
            if self.cell is not None:
                self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
            return
        if self.hidden:
            return
        if tag in {"td", "th"} and self.cell is not None:
            if self.row is not None:
                self.row.append("".join(self.cell).strip())
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    @property
    def text(self):
        return "\n".join(
            line.strip() for line in "".join(self.parts).splitlines() if line.strip()
        )


def compact(text):
    """去排版空白和括号注释后解析日期；引用仍保留提取出的原始文字。"""
    return re.sub(
        r"（[^）]*）|\([^)]*\)", "", re.sub(r"\s+", "", text.translate(TRANSLATION))
    ).replace("元月", "1月")


def date_range(text, year):
    """解析一个带月的日期或跨月范围，不推断缺失年份。"""
    found = re.search(
        r"(\d{1,2})月(\d{1,2})日?(?:至(?:(\d{1,2})月)?(\d{1,2})日)?", text
    )
    if not found:
        raise CalendarUnavailable("官方页面日期格式未通过校验。")
    month, day, end_month, end_day = found.groups()
    start = date(year, int(month), int(day))
    end = date(year, int(end_month or month), int(end_day or day))
    if end < start or (end - start).days > 15:
        raise CalendarUnavailable("官方页面出现异常日期范围。")
    return {
        (start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)
    }


def parse_notice_sections(sections, year):
    """从每个假期的正文取放假范围及补班日期，不用星期几替代官方安排。"""
    breaks, workdays = set(), set()
    for section in sections:
        text = compact(section)
        first = text.split("。", 1)[0]
        dates = date_range(first.split("放假", 1)[0], year)
        count = re.search(r"共(\d+)天", first)
        if count and int(count[1]) != len(dates):
            raise CalendarUnavailable("官方放假天数与解析日期不一致。")
        if breaks & dates:
            raise CalendarUnavailable("假期范围重复，需核实页面是否含多份通知。")
        breaks.update(dates)
        for sentence in text.split("。")[1:]:
            if "上班" in sentence:
                for month, day in re.findall(r"(\d{1,2})月(\d{1,2})日", sentence):
                    workdays.add(date(year, int(month), int(day)).isoformat())
    return breaks, workdays


def parse_sources(year, documents):
    """针对已审核网页适配；年份、栏目、日期数和交叉关系不符即拒绝。"""
    spec = SOURCE_CATALOG[year]
    notice = OfficialHTML(documents[spec["notice"]])
    policy = OfficialHTML(documents[spec["policy"]])
    if f"{year}年" not in compact(notice.text) or str(year) not in compact(policy.text):
        raise CalendarUnavailable("官方页面年份不匹配。")
    if spec["adapter"] == "table":
        if f"{year}年法定节假日工资怎么发" not in compact(notice.text):
            raise CalendarUnavailable("年度日历页面标题不匹配。")
        rows = [row for row in notice.rows if len(row) == 4]
        if not any("3倍工资待遇日期" in compact(row[2]) for row in rows):
            raise CalendarUnavailable("未找到官方三倍工资日期栏目。")
        rows = [
            row
            for row in rows
            if compact(row[0])
            in {"元旦", "春节", "清明节", "劳动节", "端午节", "国庆节中秋节"}
        ]
        if len(rows) != 6:
            raise CalendarUnavailable("年度日历表格缺少假期。")
        breaks, workdays = parse_notice_sections([row[1] for row in rows], year)
        statutory = set()
        for row in rows:
            expressions = compact(row[2])
            month = None
            for part in re.split(r"[、，,]", expressions):
                if "月" not in part:
                    if month is None:
                        raise CalendarUnavailable("法定日期缺少月份。")
                    part = f"{month}月{part}"
                month = re.search(r"(\d+)月", part)[1]
                statutory.update(date_range(part, year))
        notice_quote = "\n".join(" | ".join(row[:2]) for row in rows)
        policy_quote = "\n".join(f"{row[0]}：{row[2]}" for row in rows)
    else:
        if f"国务院办公厅关于{year}年部分节假日安排的通知" not in compact(notice.text):
            raise CalendarUnavailable("年度放假通知标题不匹配。")
        sections = re.findall(r"[一二三四五六七]、[^\n]+", notice.text)
        sections = [line for line in sections if "放假" in line]
        if len(sections) != 7:
            raise CalendarUnavailable("年度放假通知缺少七个假期段落。")
        breaks, workdays = parse_notice_sections(sections, year)
        statutory = set()
        text = re.sub(r"\s+", "", policy.text)
        if str(year) not in text or "加班工资日历版" not in text:
            raise CalendarUnavailable("官方加班工资日历的年份或标题不匹配。")
        pattern = r"(?:元旦|春节|清明节|劳动节|端午节|中秋节|国庆节)(\d{1,2}月\d{1,2}日(?:[-－—至]\d{1,2}日)?)加班[，,]另支付三倍日工资"
        entries = list(re.finditer(pattern, text))
        if len(entries) != 7:
            raise CalendarUnavailable(
                "官方PDF未完整提取七项法定节假日，不能猜测缺失日期。"
            )
        for entry in entries:
            statutory.update(date_range(re.sub(r"[-－—]", "至", entry[1]), year))
        notice_quote = "\n".join(sections)
        policy_quote = "\n".join(entry[0] for entry in entries)
    fixed = {
        f"{year}-{day}"
        for day in ("01-01", "05-01", "05-02", "10-01", "10-02", "10-03")
    }
    if (
        len(statutory) != 13
        or not fixed <= statutory
        or not statutory <= breaks
        or breaks & workdays
        or not 20 <= len(breaks) <= 40
    ):
        raise CalendarUnavailable("日历交叉校验失败，不能据此计算工资。")
    return {
        "year": year,
        "statutory": sorted(statutory),
        "breaks": sorted(breaks),
        "workdays": sorted(workdays),
        "sources": [
            {
                "source_id": f"官方日历#{year}年放假安排",
                "kind": "official_calendar",
                "url": spec["notice"],
                "text": notice_quote,
            },
            {
                "source_id": f"官方说明#{year}年法定节假日",
                "kind": "official_policy",
                "url": spec["policy"],
                "text": policy_quote,
            },
        ],
    }


def check_url(url):
    """固定官方域名、HTTPS、标准端口；重定向也不能跳到内网或任意网站。"""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.port not in {None, 443}
        or parsed.username
        or parsed.password
    ):
        raise CalendarUnavailable("日历来源不在允许的官方站点中。")


def download_official(url):
    """限制重定向、下载大小和等待时间；网页内容不进入模型指令。"""
    load_env()
    with httpx.Client(
        timeout=httpx.Timeout(8, connect=5),
        trust_env=True,
        proxy=os.getenv("HOLIDAY_HTTP_PROXY") or None,
    ) as client:
        started = time.monotonic()
        for _ in range(3):
            check_url(url)
            with client.stream(
                "GET",
                url,
                headers={"User-Agent": "LaborLawCalendar/1.0", "Accept": "text/html"},
            ) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get("location", ""))
                    continue
                response.raise_for_status()
                is_pdf = urlsplit(url).path.lower().endswith(".pdf")
                expected_type = "application/pdf" if is_pdf else "text/html"
                if (
                    expected_type
                    not in response.headers.get("content-type", "").lower()
                ):
                    raise CalendarUnavailable("官方来源未返回所需的HTML或PDF正文。")
                payload = bytearray()
                for chunk in response.iter_bytes():
                    payload.extend(chunk)
                    if len(payload) > MAX_BYTES or time.monotonic() - started > 15:
                        raise CalendarUnavailable("日历下载超过大小或耗时限制。")
                if is_pdf:
                    from pypdf import PdfReader  # 仅PDF来源使用轻量解析库。

                    try:
                        reader = PdfReader(BytesIO(payload), strict=True)
                        if reader.is_encrypted or len(reader.pages) > 4:
                            raise CalendarUnavailable("日历PDF加密或页数异常。")
                        text = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        )
                        if len(text) > 100000:
                            raise CalendarUnavailable("日历PDF文字过大。")
                        return f"<pre>{escape(text)}</pre>"
                    except CalendarUnavailable:
                        raise
                    except Exception as error:
                        raise CalendarUnavailable(
                            "官方PDF无法安全提取文本。"
                        ) from error
                return bytes(payload).decode("utf-8-sig", errors="strict")
    raise CalendarUnavailable("官方页面重定向过多。")


class OfficialCalendarProvider:
    """先使用24小时内的已校验缓存；过期/损坏时刷新，失败则停止计算。"""

    def __init__(self, cache_dir=None, fetcher=download_official, clock=None):
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else PROJECT_ROOT / "data" / "holiday_calendar"
        )
        self.fetcher = fetcher
        self.clock = clock or (lambda: datetime.now(UTC))
        self.lock = RLock()  # 单进程内相同请求不并发覆盖缓存文件。

    def _validate_cache(self, year, item):
        spec = SOURCE_CATALOG[year]
        if (
            item["version"] != 1
            or item["year"] != year
            or set(item["documents"]) != {spec["notice"], spec["policy"]}
        ):
            raise CalendarUnavailable("缓存版本、年份或来源不符。")
        fetched = datetime.fromisoformat(item["fetched_at"])
        if (
            fetched.tzinfo is None
            or not 0 <= (self.clock() - fetched).total_seconds() < CACHE_TTL_SECONDS
        ):
            raise CalendarUnavailable("缓存过期或时间无效。")
        result = parse_sources(
            year, item["documents"]
        )  # 不信缓存内的现成日期标签，重新解析原网页。
        result.update(
            {
                "fetched_at": fetched.isoformat(),
                "expires_at": (
                    fetched + timedelta(seconds=CACHE_TTL_SECONDS)
                ).isoformat(),
            }
        )
        return result

    def get(self, year, refresh=False):
        if year not in SOURCE_CATALOG:
            raise CalendarUnavailable(
                "该年份尚未接入经过审核的官方来源（当前2025、2026）；不代表官方未公布，请补充可靠来源，不能套用其他年份。"
            )
        with self.lock:
            path = self.cache_dir / f"{year}.json"
            if not refresh:
                try:
                    if path.stat().st_size > 3 * MAX_BYTES:
                        raise CalendarUnavailable("缓存过大。")
                    result = self._validate_cache(
                        year, json.loads(path.read_text(encoding="utf-8"))
                    )
                    return {**result, "data_origin": "cache", "stale": False}
                except (OSError, ValueError, KeyError, TypeError, IndexError):
                    pass  # 缓存无效时只允许真实刷新，不能退回旧的硬编码日历。
            try:
                spec = SOURCE_CATALOG[year]
                documents = {
                    url: self.fetcher(url) for url in {spec["notice"], spec["policy"]}
                }
                item = {
                    "version": 1,
                    "year": year,
                    "fetched_at": self.clock().isoformat(),
                    "documents": documents,
                }
                result = self._validate_cache(year, item)
            except (
                httpx.HTTPError,
                OSError,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
                StopIteration,
            ) as error:
                raise CalendarUnavailable(
                    "无法获取并校验官方日历，未使用过期或旧内置数据。请检查网络、代理或页面格式后重试。"
                ) from error
            cache_saved = False
            temporary = None
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.cache_dir,
                    suffix=".tmp",
                    delete=False,
                ) as target:
                    temporary = target.name
                    json.dump(item, target, ensure_ascii=False)
                os.replace(temporary, path)
                cache_saved = True
            except OSError:
                pass  # 写缓存失败不丢弃本次已联网核验的数据，但对外如实标注。
            finally:
                if temporary and os.path.exists(temporary):
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass
            return {
                **result,
                "data_origin": "network",
                "stale": False,
                "cache_saved": cache_saved,
            }


@lru_cache(maxsize=1)
def default_calendar_provider():
    return OfficialCalendarProvider()  # 仅在首次工具请求时初始化，不在服务启动时联网。
