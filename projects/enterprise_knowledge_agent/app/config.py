"""集中读取 DeepSeek 配置，避免在业务代码中散落环境变量。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEARNING_ROOT = PROJECT_ROOT.parent.parent

# 检索门控阈值：低于该相似度的片段一律视为“知识库没有答案”。
# 0.5 在只有 4 段玩具数据时够用，但换成 107 条真实法条后，域外问题
# 的相似度普遍落在 0.5~0.67，实测误放行率高达 58%；
# 上调到 0.6 后误放行率降到 8%，同时域内问题的最高分仍全部通过门控。
#
# 加入《劳动合同法》后（合计 205 条）重新验证过，阈值没有改：
# 域内 18 条 Top-1 78%、证据召回 94%、误拒答 0；域外 13 条正确拒答 69%。
# 拒答率比单法源时的 77% 低，原因是库里有更多“擦边相关内容”会被放行
# （如竞业限制），属于知识库变大后的必然代价，不是门控失效。
# 阈值集中在这里，避免各处硬编码。
KNOWLEDGE_MIN_SCORE = 0.6

# 返回给模型的最大条文数。这个值比阈值更容易被忽略，但影响很大：
# “法定节假日加班工资怎么算”的正确答案是第四十四条，得分 0.6216，
# 排在第五十一条(0.6519)、第四十条(0.6267) 之后。top_k=2 时它被挤掉，
# 模型只看到“应依法支付工资”，于是如实回答“无法确定倍数”——这不是
# 模型幻觉，而是证据没喂到。实测 top_k=2→4 让“正确条文进入证据”的
# 比例从 83% 提升到 89%，而域外问题的误放行没有增加。
KNOWLEDGE_TOP_K = 4

# ---------------------------------------------------------------------------
# 混合检索（向量 + BM25 关键词）相关参数
# ---------------------------------------------------------------------------

# 关键词通道的门控阈值：查询词在这条法条上的 idf 加权覆盖率（0~1）。
# 它补的是向量通道的短板：向量把“书面形式”和“合同变更”算得很接近，
# 但覆盖率能直接回答“这些字到底有没有出现在这条法条里”。
# 0.15 是在 18 条域内 + 12 条域外问题上扫描出来的：低于它救不回
# “法定节假日加班给几倍工资”（向量 0.58 过不了 0.6），高于 0.2 则会把
# “竞业限制补偿金”这类域外问题放行。
KNOWLEDGE_KEYWORD_COVERAGE = 0.15

# 两路分数的融合权重（等权）。实测在 0.3~0.6 之间调整，Top-1 命中率
# 没有任何变化，说明融合结果对权重不敏感，因此选等权，最容易解释。
KNOWLEDGE_VECTOR_WEIGHT = 0.5
KNOWLEDGE_KEYWORD_WEIGHT = 0.5

# BM25 分数没有固定上界，直接和余弦相似度相加会被它主导。
# 用 b/(b+K) 把 BM25 分数压到 0~1：域内问题常见 4~11 分，域外常见 0~8 分，
# K=5 能让两者落在同一量纲，且是单调映射，不改变单路排序。
KNOWLEDGE_BM25_SCALE = 5.0


class Settings(BaseModel):
    """应用启动所需的最小配置。"""

    api_key: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)


def load_env() -> None:
    """把项目 .env 读进进程环境；重复调用是幂等的。

    单独抽出来是因为不止 ``load_settings`` 需要它：认证模块也要读环境变量，
    但它不关心 DeepSeek 的 Key 是否存在，不该为了读一个令牌就要求配好模型。
    """
    project_env = PROJECT_ROOT / ".env"
    learning_env = LEARNING_ROOT / ".env"
    load_dotenv(project_env if project_env.exists() else learning_env)


def load_settings() -> Settings:
    """优先读取项目 .env，并兼容当前学习仓库根目录中的 .env。"""
    load_env()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY，请配置项目 .env。")

    return Settings(
        api_key=api_key,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        # 允许通过 DEEPSEEK_BASE_URL 切换到任何 OpenAI 兼容服务
        # （如硅基流动、阿里百炼、本地 Ollama）。换供应商只改 .env，
        # 不动代码，因为项目本身只依赖 OpenAI 兼容协议。
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
