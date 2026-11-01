"""对检索层做泛化验证：同时检查“该检到的有没有检到”和“不该答的有没有拒”。

和 `evaluate_retrieval.py` 的区别：
- 评估集（evals/*.json）是少量人工标注样本，用来做质量门槛；
- 本脚本用更大一批自然问法做诊断，包含已用于调参的问题，不能当作独立 holdout；重点是发现
  换成真实法条后阈值门控是否仍然可靠。

不调用 DeepSeek，只用本地向量索引，因此零成本、可反复运行。
"""

from projects.enterprise_knowledge_agent.app.config import (
    KNOWLEDGE_KEYWORD_COVERAGE,
    KNOWLEDGE_MIN_SCORE,
    KNOWLEDGE_TOP_K,
)
from projects.enterprise_knowledge_agent.app.retrieval import VectorKnowledgeBase
from projects.enterprise_knowledge_agent.app.tools import ToolRegistry


# 域内问题：可接受的来源集合，格式固定为“法律简称#条号”。
# 用集合而不是单一编号，是因为同一问题常有多个同样正确的落点：
# “拖欠工资怎么办”既可以引劳动法第五十条（不得拖欠），也可以引第九十一条（拖欠后的处置）。
#
# 加入《劳动合同法》后，标识必须带法律名。两部法律都有“第一条”“第四十四条”，
# 只用条号做子串匹配会让“劳动合同法#第四十四条”也算命中“第四十四条”，
# 基准从此不可信。精确匹配（==）比子串匹配更适合这里：
# 宁可显式列出每一个可接受的来源，也不要靠字符串包含蒙对。
IN_SCOPE_CASES: list[tuple[str, tuple[str, ...]]] = [
    ("加班工资按什么标准支付", ("劳动法#第四十四条",)),
    ("劳动者每天工作时间最长几小时", ("劳动法#第三十六条",)),
    # 两部法律都规定了试用期上限，劳动合同法第十九条更细（按合同期限分档）。
    ("试用期最长可以多久", ("劳动法#第二十一条", "劳动合同法#第十九条")),
    ("产假最少多少天", ("劳动法#第六十二条",)),
    # 劳动法第三十一条只写三十日；劳动合同法第三十七条补了试用期三日。
    (
        "辞职需要提前多久通知单位",
        ("劳动法#第三十一条", "劳动合同法#第三十七条"),
    ),
    ("最低工资由谁确定", ("劳动法#第四十八条",)),
    # 注意第三十八条是“劳动者可以解除”，方向相反，不能算对。
    (
        "什么情况下用人单位可以解除劳动合同",
        (
            "劳动法#第二十五条",
            "劳动法#第二十六条",
            "劳动合同法#第三十九条",
            "劳动合同法#第四十条",
        ),
    ),
    # 劳动法第二十八条只说“依照国家有关规定给予补偿”，没给算法；
    # 真正的计算标准在劳动合同法第四十七条。旧标注只有第二十八条，已过时。
    ("经济补偿金怎么计算", ("劳动合同法#第四十七条", "劳动法#第二十八条")),
    ("女职工在经期有什么保护", ("劳动法#第六十条",)),
    ("未成年工不能从事哪些劳动", ("劳动法#第六十四条",)),
    (
        "劳动合同必须书面订立吗",
        ("劳动法#第十九条", "劳动合同法#第十条"),
    ),
    ("每月最多能加班多少小时", ("劳动法#第四十一条",)),
    ("法定节假日加班给几倍工资", ("劳动法#第四十四条",)),
    ("法定节假日加班工资怎么算？", ("劳动法#第四十四条",)),
    ("劳动争议可以直接起诉吗", ("劳动法#第七十九条",)),
    ("养老保险和医疗保险由谁缴纳", ("劳动法#第七十二条",)),
    (
        "用人单位拖欠工资怎么办",
        ("劳动法#第九十一条", "劳动法#第五十条", "劳动合同法#第三十条"),
    ),
    # 新增法条带来的能力：竞业限制的期限只有劳动合同法写了，劳动法没有。
    ("竞业限制最多几年", ("劳动合同法#第二十四条",)),
]

# 域外问题：知识库里没有答案，正确行为是拒答而不是硬凑一段法条。
OUT_OF_SCOPE_CASES: list[str] = [
    "公司年会抽奖的奖品怎么发放",
    "差旅费报销需要哪些材料",
    # 加入《劳动合同法》后，这条的处境变了：第二十三条确实规定了竞业限制
    # 应当按月给予经济补偿，但**没有规定具体标准**（数额在司法解释和地方
    # 实践中）。所以它仍是域外题——问的是“标准是多少”，不是“有没有”。
    # 检索层现在必然会放行（库里有竞业限制条款），守住它的是生成层。
    "竞业限制补偿金标准是多少",
    "公司股权激励怎么分配",
    "员工宿舍分配规则",
    "怎么申请办公电脑",
    "年终奖发几个月",
    "社保缴费基数怎么核定",
    "请客户吃饭的预算标准",
    "公司股票什么时候上市",
    "公积金贷款额度怎么算",
    "打卡迟到扣多少钱",
    # 医疗费的具体承担属于《工伤保险条例》，劳动法只写了“依法享受社会保险
    # 待遇”，没有写谁付钱。之前把它标成域内问题是标注错误，会让检索层
    # 背不属于它的锅。
    "工伤医疗费由谁承担",
]


def describe_channels(match: dict[str, object]) -> str:
    """说明这条结果是被哪条门控放行的，便于观察两路信号各自的贡献。"""
    by_vector = float(match["vector_score"]) >= KNOWLEDGE_MIN_SCORE
    by_keyword = float(match["keyword_coverage"]) >= KNOWLEDGE_KEYWORD_COVERAGE
    if by_vector and by_keyword:
        return "向量+关键词"
    return "向量" if by_vector else "仅关键词"


def main() -> None:
    """打印检索命中情况与门控误放行率，便于对比不同阈值的取舍。"""
    knowledge_base = VectorKnowledgeBase.from_disk()
    registry = ToolRegistry(knowledge_base)  # 直接复用生产阈值，保证验证和线上一致。
    print(
        f"当前门控：向量相似度 >= {KNOWLEDGE_MIN_SCORE} "
        f"或 关键词覆盖率 >= {KNOWLEDGE_KEYWORD_COVERAGE}；Top-K：{KNOWLEDGE_TOP_K}\n"
    )

    hit_at_1 = 0
    hit_at_k = 0
    refused_by_gate = 0
    rescued_by_keyword: list[str] = []
    for question, acceptable_articles in IN_SCOPE_CASES:
        matches = registry.search_knowledge_base({"query": question})["matches"]
        sources = [str(match["source_id"]) for match in matches]

        def acceptable(source: str) -> bool:
            # 精确相等，不做子串匹配：source 形如“劳动法#第四十四条”。
            return source in acceptable_articles

        # 记录“向量通道本来会挡掉、靠关键词通道救回来”的样本，这是混合检索的核心收益。
        rescue = [
            match
            for match in matches
            if acceptable(str(match["source_id"]))
            and float(match["vector_score"]) < KNOWLEDGE_MIN_SCORE
        ]
        for match in rescue:
            rescued_by_keyword.append(
                f"{question} -> {match['source_id']}"
                f"（向量 {float(match['vector_score']):.3f} 未过线，"
                f"关键词覆盖率 {float(match['keyword_coverage']):.3f}）"
            )

        if not sources:
            refused_by_gate += 1  # 域内问题被门控挡掉，属于“误拒答”。
            print(f"误拒答  {question}（可接受 {'/'.join(acceptable_articles)}）")
            continue

        if acceptable(sources[0]):
            hit_at_1 += 1
        elif any(acceptable(source) for source in sources):
            hit_at_k += 1
            print(f"第K位   {question} -> {sources[0]}")
        else:
            print(f"未召回  {question}（可接受 {'/'.join(acceptable_articles)}，实际 {sources}）")

    total_in_scope = len(IN_SCOPE_CASES)
    correct_refusals = 0
    leaked: list[str] = []
    for question in OUT_OF_SCOPE_CASES:
        matches = registry.search_knowledge_base({"query": question})["matches"]
        if matches:
            top = matches[0]
            leaked.append(
                f"{question} -> {top['source_id']} "
                f"[{describe_channels(top)}] 融合分 {float(top['score']):.3f}"
            )
        else:
            correct_refusals += 1

    print()
    print(f"域内 Top-1 命中率：{hit_at_1}/{total_in_scope} = {hit_at_1 / total_in_scope:.0%}")
    print(
        f"域内证据召回（正确条文进入前 {KNOWLEDGE_TOP_K} 条）："
        f"{hit_at_1 + hit_at_k}/{total_in_scope} = {(hit_at_1 + hit_at_k) / total_in_scope:.0%}"
    )
    print("  说明：Top-1 衡量检索精度，证据召回衡量“模型有没有机会看到正确答案”。")
    print("        只看 Top-1 会误判——Top-1 命中七成时，证据召回仍可能接近满分。")
    print(f"域内误拒答：{refused_by_gate}/{total_in_scope}")
    print(
        f"域外正确拒答：{correct_refusals}/{len(OUT_OF_SCOPE_CASES)} = "
        f"{correct_refusals / len(OUT_OF_SCOPE_CASES):.0%}"
    )
    if leaked:
        print("门控放行的域外问题（模型仍可能拒答，但属于检索层漏检）：")
        for item in leaked:
            print(f"  - {item}")
    if rescued_by_keyword:
        print("靠关键词通道救回的正确条文（纯向量会漏掉）：")
        for item in rescued_by_keyword:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
