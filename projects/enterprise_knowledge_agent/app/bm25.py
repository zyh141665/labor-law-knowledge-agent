"""纯 Python 实现的 BM25 关键词检索，用来补足向量召回对术语不敏感的问题。

为什么不用 jieba 之类的分词器：

1. 法条里全是“法定休假日”“用人单位”“经济补偿”这类专业术语，通用词典
   经常切错，切错就直接导致关键词匹配失败；
2. 中文用**字符 bigram** 对术语已经足够敏感：“法定休假日”和“法定节假日”
   虽然整体不同，但共享“法定”“假日”两个 token，正好覆盖了同义换写的场景；
3. 零依赖，索引可以随时从条文原文重建，不需要额外的词典文件。

BM25 与向量召回的差别在于：向量看“语义像不像”，BM25 看“这些字有没有真的
出现在原文里”。法条这种短文本、术语密集的语料，后者往往更准。
"""

import math
import re
from collections import Counter


CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
ALNUM_RUN = re.compile(r"[a-zA-Z0-9]+")

# 中文疑问句里的高频虚词没有检索价值，而且它们往往是法条中不存在的词。
# 如果不剔除，这些词会被算成“极其罕见的词”（idf 很高），既干扰 BM25 打分，
# 也会把关键词覆盖率的分母撑大，导致正确条文被误判成“不相关”。
STOPWORDS = (
    "怎么",
    "怎样",
    "如何",
    "什么",
    "哪些",
    "哪个",
    "多少",
    "是否",
    "能否",
    "可以",
    "需要",
    "应该",
    "请问",
    "一下",
)


def normalize(text: str) -> str:
    """去掉没有检索价值的疑问词，避免它们污染 token 和覆盖率分母。"""
    for word in STOPWORDS:
        text = text.replace(word, " ")
    return text


def tokenize(text: str) -> list[str]:
    """把中文切成字符 bigram，数字和英文按整词保留。

    单字成句的中文（例如“法”）保留为单 token，否则会被切空。
    """
    tokens: list[str] = []
    for run in CJK_RUN.findall(normalize(text)):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    tokens.extend(token.lower() for token in ALNUM_RUN.findall(text))
    return tokens


class BM25Index:
    """在一批文档上计算 BM25 分数和关键词覆盖率。

    - `score()`：标准 BM25，衡量“查询词在文档里出现得多不多”；
    - `coverage()`：idf 加权的覆盖率，衡量“查询里的关键词有多大比例真的
      出现在这篇文档里”，取值 0~1，可以直接当门控阈值用。

    之所以单独提供 coverage：BM25 原始分没有固定上限；若选择按每次查询
    最大值归一化，非零最高分总是 1，不适合绝对门控。当前排序使用的是
    b/(b+5)，并非最大值归一化。覆盖率也是经验信号，不保证相关文档能回答问题。
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25 索引至少需要一篇文档。")
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(document)) for document in documents]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)

        document_frequency: Counter[str] = Counter()
        for counter in self.term_frequencies:
            document_frequency.update(counter.keys())
        total = len(documents)
        # 标准 BM25 idf；df=0 时给出上界，避免完全没出现过的词把分母撑爆。
        self.idf = {
            token: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.unseen_idf = math.log(1 + (total + 0.5) / 0.5)

    def _idf(self, token: str) -> float:
        return self.idf.get(token, self.unseen_idf)

    def score(self, query: str) -> list[float]:
        """返回查询对每篇文档的 BM25 分数。"""
        terms = set(tokenize(query))
        if not terms:
            return [0.0] * len(self.term_frequencies)

        scores: list[float] = []
        for counter, length in zip(self.term_frequencies, self.lengths):
            total = 0.0
            for term in terms:
                frequency = counter.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                total += self._idf(term) * frequency * (self.k1 + 1) / denominator
            scores.append(total)
        return scores

    def coverage(self, query: str) -> list[float]:
        """返回每篇文档对查询关键词的 idf 加权覆盖率（0~1）。

        分母包含知识库里从未出现过的词，所以问“知识库根本没有的东西”时，
        分母会很大、覆盖率会很低——这正是门控需要的信号。
        """
        terms = set(tokenize(query))
        if not terms:
            return [0.0] * len(self.term_frequencies)

        total_weight = sum(self._idf(term) for term in terms)
        if total_weight == 0:
            return [0.0] * len(self.term_frequencies)

        coverages: list[float] = []
        for counter in self.term_frequencies:
            matched = sum(
                self._idf(term) for term in terms if counter.get(term, 0)
            )
            coverages.append(matched / total_weight)
        return coverages
