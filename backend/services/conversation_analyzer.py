"""
对话分析器模块 - 文档摘要核心功能

提供对话历史分析、文档摘要生成、主题提取等功能。
采用模块化设计，支持多种摘要算法和分析策略。
"""

import re
import math
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Callable
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import jieba
import heapq

from backend.models.llm_model import get_llm_model
from backend.services.conversation_store import (
    ConversationSession, ConversationTurn, get_conversation_store
)


class SummaryStrategy(Enum):
    """摘要生成策略枚举"""
    EXTRACTIVE = "extractive"  # 抽取式摘要
    ABSTRACTIVE = "abstractive"  # 生成式摘要
    HYBRID = "hybrid"  # 混合式摘要


class SummaryLength(Enum):
    """摘要长度控制"""
    SHORT = 0.15  # 简短摘要（原文15%）
    MEDIUM = 0.30  # 中等摘要（原文30%）
    LONG = 0.50  # 详细摘要（原文50%）


@dataclass
class SummaryResult:
    """摘要结果数据结构"""
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_points: List[str]
    keywords: List[str]
    strategy: str
    topic: str = ""


@dataclass
class DocumentChunk:
    """文档分块数据结构"""
    id: int
    text: str
    sentences: List[str]
    word_count: int
    importance_score: float = 0.0


@dataclass
class TopicSegment:
    """主题段落数据结构"""
    start_idx: int
    end_idx: int
    topic: str
    sentences: List[str]
    keywords: List[str]


class TextPreprocessor:
    """文本预处理器 - 提供文本清洗、分词、分句等基础功能"""

    # 中文停用词表
    _stopwords = {
        '的', '了', '和', '是', '在', '我', '有', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
        '自己', '这', '下面', '还有', '这个', '那个', '这些', '那些', '但是', '而且',
        '不过', '然后', '所以', '因为', '所以', '可以', '能够', '可能', '应该', '必须'
    }

    # 分句标点符号
    _sentence_delimiters = r'[。！？；.!?;]'

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清洗文本，去除冗余字符和格式

        Args:
            text: 原始文本

        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""

        # 替换特殊空白字符
        text = text.replace('\u3000', ' ').replace('\xa0', ' ')

        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 去除多余的空白字符
        text = re.sub(r'[ \t]+', ' ', text)

        # 合并多个空行
        text = re.sub(r'\n\s*\n', '\n', text)

        # 去除首尾空白
        text = text.strip()

        return text

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        将文本分割为句子列表

        Args:
            text: 输入文本

        Returns:
            List[str]: 句子列表
        """
        if not text:
            return []

        # 清洗文本
        text = TextPreprocessor.clean_text(text)

        # 分句
        sentences = re.split(TextPreprocessor._sentence_delimiters, text)

        # 过滤空句子并添加标点
        processed = []
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if sent:
                # 尝试恢复原标点
                if i < len(sentences) - 1:
                    processed.append(sent + '。')
                else:
                    processed.append(sent)

        return processed

    @staticmethod
    def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
        """
        中文分词

        Args:
            text: 输入文本
            remove_stopwords: 是否移除停用词

        Returns:
            List[str]: 分词结果
        """
        if not text:
            return []

        # 使用jieba分词
        words = jieba.lcut(text, cut_all=False)

        # 过滤和清洗
        words = [w.strip() for w in words if w.strip()]

        if remove_stopwords:
            words = [w for w in words if w not in TextPreprocessor._stopwords]

        return words

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """
        提取关键词（基于词频）

        Args:
            text: 输入文本
            top_n: 返回关键词数量

        Returns:
            List[str]: 关键词列表
        """
        words = TextPreprocessor.tokenize(text)

        # 过滤单字符词
        words = [w for w in words if len(w) > 1]

        # 统计词频
        word_counts = Counter(words)

        # 返回Top-N关键词
        return [word for word, _ in word_counts.most_common(top_n)]

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（基于Jaccard系数）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 相似度分数 [0, 1]
        """
        words1 = set(TextPreprocessor.tokenize(text1))
        words2 = set(TextPreprocessor.tokenize(text2))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)


class ISummarizer(ABC):
    """摘要生成器接口"""

    @abstractmethod
    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        ratio: float = 0.3
    ) -> SummaryResult:
        """
        生成文本摘要

        Args:
            text: 输入文本
            max_length: 最大长度（字符数）
            ratio: 压缩比例（相对于原文）

        Returns:
            SummaryResult: 摘要结果
        """
        pass


class ExtractiveSummarizer(ISummarizer):
    """
    抽取式摘要生成器

    基于TextRank算法，通过句子之间的相似度计算句子重要性，
    选取最重要的句子组成摘要。
    """

    def __init__(self, damping_factor: float = 0.85, max_iter: int = 50):
        """
        初始化抽取式摘要器

        Args:
            damping_factor: TextRank阻尼系数
            max_iter: 最大迭代次数
        """
        self.damping_factor = damping_factor
        self.max_iter = max_iter

    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        构建句子相似度矩阵

        Args:
            sentences: 句子列表

        Returns:
            np.ndarray: 相似度矩阵
        """
        n = len(sentences)
        if n == 0:
            return np.array([])

        # 初始化相似度矩阵
        sim_matrix = np.zeros((n, n))

        # 计算两两相似度
        for i in range(n):
            for j in range(i + 1, n):
                sim = TextPreprocessor.calculate_similarity(sentences[i], sentences[j])
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        # 归一化（按行求和）
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除以0
        sim_matrix = sim_matrix / row_sums

        return sim_matrix

    def _calculate_text_rank(self, sim_matrix: np.ndarray) -> np.ndarray:
        """
        使用TextRank算法计算句子得分

        Args:
            sim_matrix: 相似度矩阵

        Returns:
            np.ndarray: 句子得分数组
        """
        n = len(sim_matrix)
        if n == 0:
            return np.array([])

        # 初始化得分
        scores = np.ones(n) / n

        # 迭代计算
        for _ in range(self.max_iter):
            new_scores = (1 - self.damping_factor) + self.damping_factor * sim_matrix.T.dot(scores)

            # 检查收敛
            if np.sum(np.abs(new_scores - scores)) < 1e-6:
                break

            scores = new_scores

        return scores

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        ratio: float = 0.3
    ) -> SummaryResult:
        """
        生成抽取式摘要

        Args:
            text: 输入文本
            max_length: 最大字符长度
            ratio: 压缩比例（相对于原文句子数）

        Returns:
            SummaryResult: 摘要结果
        """
        original_text = TextPreprocessor.clean_text(text)
        sentences = TextPreprocessor.split_sentences(original_text)

        if not sentences:
            return SummaryResult(
                summary="",
                original_length=len(original_text),
                summary_length=0,
                compression_ratio=0.0,
                key_points=[],
                keywords=[],
                strategy=SummaryStrategy.EXTRACTIVE.value
            )

        # 构建相似度矩阵并计算得分
        sim_matrix = self._build_similarity_matrix(sentences)
        scores = self._calculate_text_rank(sim_matrix)

        # 确定摘要长度
        num_sentences = max(1, int(len(sentences) * ratio))
        if max_length:
            # 按字符数限制调整
            current_length = 0
            selected_indices = []
            # 按得分排序
            ranked_indices = np.argsort(scores)[::-1]

            for idx in ranked_indices:
                sent_length = len(sentences[idx])
                if current_length + sent_length <= max_length:
                    selected_indices.append(idx)
                    current_length += sent_length

            selected_indices.sort()
        else:
            # 选择Top-N个得分最高的句子（保持原文顺序）
            top_indices = np.argsort(scores)[-num_sentences:]
            selected_indices = sorted(top_indices)

        # 生成摘要
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = ''.join(summary_sentences)

        # 提取关键点和关键词
        key_points = [sentences[i] for i in sorted(np.argsort(scores)[-5:])]
        keywords = TextPreprocessor.extract_keywords(original_text)

        # 推断主题
        topic = self._infer_topic(keywords, key_points)

        return SummaryResult(
            summary=summary,
            original_length=len(original_text),
            summary_length=len(summary),
            compression_ratio=round(len(summary) / len(original_text) if len(original_text) > 0 else 0, 3),
            key_points=key_points,
            keywords=keywords,
            strategy=SummaryStrategy.EXTRACTIVE.value,
            topic=topic
        )

    def _infer_topic(self, keywords: List[str], key_points: List[str]) -> str:
        """
        推断文本主题

        Args:
            keywords: 关键词列表
            key_points: 关键句子列表

        Returns:
            str: 主题描述
        """
        if keywords:
            return keywords[0]
        elif key_points:
            # 返回第一个关键点的前20字符
            return key_points[0][:20] + "..."
        return "未分类主题"


class AbstractiveSummarizer(ISummarizer):
    """
    生成式摘要生成器

    使用LLM模型生成更自然、更具概括性的摘要，
    支持多文档摘要和对话摘要。
    """

    def __init__(self):
        """初始化生成式摘要器"""
        self.llm = get_llm_model()

    def _build_prompt(
        self,
        text: str,
        max_length: Optional[int] = None,
        focus_points: Optional[List[str]] = None
    ) -> str:
        """
        构建摘要生成prompt

        Args:
            text: 待摘要文本
            max_length: 最大长度
            focus_points: 需要重点关注的要点

        Returns:
            str: 生成的prompt
        """
        length_constraint = f"，摘要长度不超过{max_length}字" if max_length else ""
        focus_constraint = f"，重点关注以下内容：{'; '.join(focus_points)}" if focus_points else ""

        prompt = f"""请对以下文本生成一段简洁、准确的中文摘要{length_constraint}{focus_constraint}。

要求：
1. 保留核心信息和关键观点
2. 语言流畅、自然
3. 结构清晰，逻辑连贯
4. 避免冗余信息

待摘要文本：
{text}

摘要："""

        return prompt

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        ratio: float = 0.3
    ) -> SummaryResult:
        """
        生成生成式摘要

        Args:
            text: 输入文本
            max_length: 最大字符长度
            ratio: 压缩比例（如未指定max_length时使用）

        Returns:
            SummaryResult: 摘要结果
        """
        original_text = TextPreprocessor.clean_text(text)

        if not original_text:
            return SummaryResult(
                summary="",
                original_length=0,
                summary_length=0,
                compression_ratio=0.0,
                key_points=[],
                keywords=[],
                strategy=SummaryStrategy.ABSTRACTIVE.value
            )

        # 计算目标长度
        if not max_length:
            target_length = int(len(original_text) * ratio)
            max_length = min(target_length, 1000)  # 限制最大长度

        # 构建prompt并生成摘要
        prompt = self._build_prompt(original_text, max_length)
        summary = self.llm.generate_summary(original_text, max_length=max_length)

        # 提取关键词和关键点
        keywords = TextPreprocessor.extract_keywords(original_text)
        key_points = self._extract_key_points(original_text)

        return SummaryResult(
            summary=summary,
            original_length=len(original_text),
            summary_length=len(summary),
            compression_ratio=round(len(summary) / len(original_text) if len(original_text) > 0 else 0, 3),
            key_points=key_points,
            keywords=keywords,
            strategy=SummaryStrategy.ABSTRACTIVE.value,
            topic=keywords[0] if keywords else ""
        )

    def _extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        从文本中提取关键点

        Args:
            text: 输入文本
            num_points: 提取数量

        Returns:
            List[str]: 关键点列表
        """
        sentences = TextPreprocessor.split_sentences(text)
        if len(sentences) <= num_points:
            return sentences

        # 使用TextRank选择最重要的句子
        summarizer = ExtractiveSummarizer()
        result = summarizer.summarize(text, ratio=0.3)
        return result.key_points[:num_points]


class HybridSummarizer(ISummarizer):
    """
    混合式摘要生成器

    结合抽取式和生成式方法：
    1. 先用抽取式方法提取关键句子
    2. 再用生成式方法进行润色和重组
    """

    def __init__(self):
        """初始化混合式摘要器"""
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer()

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        ratio: float = 0.3
    ) -> SummaryResult:
        """
        生成混合式摘要

        Args:
            text: 输入文本
            max_length: 最大字符长度
            ratio: 压缩比例

        Returns:
            SummaryResult: 摘要结果
        """
        # 第一步：抽取式摘要（抽取50%作为候选）
        extractive_result = self.extractive.summarize(text, ratio=min(ratio * 1.5, 0.6))

        if not extractive_result.summary:
            return extractive_result

        # 第二步：生成式润色
        polish_prompt = f"""请对以下内容进行润色和重组，生成一段流畅的摘要{
            f'，不超过{max_length}字' if max_length else ''}。
要求保留所有关键信息，语言要自然流畅：

{extractive_result.summary}

润色后的摘要："""

        polished_summary = self.abstractive.llm.generate_summary(
            polish_prompt,
            max_length=max_length or len(extractive_result.summary)
        )

        return SummaryResult(
            summary=polished_summary,
            original_length=extractive_result.original_length,
            summary_length=len(polished_summary),
            compression_ratio=round(len(polished_summary) / extractive_result.original_length, 3),
            key_points=extractive_result.key_points,
            keywords=extractive_result.keywords,
            strategy=SummaryStrategy.HYBRID.value,
            topic=extractive_result.topic
        )


class SummaryFactory:
    """摘要生成器工厂"""

    _creators: Dict[SummaryStrategy, Callable[[], ISummarizer]] = {
        SummaryStrategy.EXTRACTIVE: lambda: ExtractiveSummarizer(),
        SummaryStrategy.ABSTRACTIVE: lambda: AbstractiveSummarizer(),
        SummaryStrategy.HYBRID: lambda: HybridSummarizer()
    }

    @staticmethod
    def create_summarizer(strategy: SummaryStrategy) -> ISummarizer:
        """
        创建摘要生成器

        Args:
            strategy: 摘要策略

        Returns:
            ISummarizer: 摘要生成器实例
        """
        creator = SummaryFactory._creators.get(strategy)
        if not creator:
            raise ValueError(f"不支持的摘要策略: {strategy}")
        return creator()


class ConversationAnalyzer:
    """
    对话分析器 - 提供对话历史分析和摘要功能

    整合多种分析能力，支持单轮对话分析、多轮对话摘要、主题分析等。
    """

    def __init__(
        self,
        default_strategy: SummaryStrategy = SummaryStrategy.HYBRID,
        default_ratio: float = SummaryLength.MEDIUM.value
    ):
        """
        初始化对话分析器

        Args:
            default_strategy: 默认摘要策略
            default_ratio: 默认压缩比例
        """
        self.default_strategy = default_strategy
        self.default_ratio = default_ratio
        self.summarizers: Dict[SummaryStrategy, ISummarizer] = {}
        self._initialize_summarizers()

    def _initialize_summarizers(self) -> None:
        """初始化所有摘要生成器"""
        for strategy in SummaryStrategy:
            self.summarizers[strategy] = SummaryFactory.create_summarizer(strategy)

    def _get_summarizer(self, strategy: SummaryStrategy) -> ISummarizer:
        """获取指定策略的摘要生成器"""
        if strategy not in self.summarizers:
            self.summarizers[strategy] = SummaryFactory.create_summarizer(strategy)
        return self.summarizers[strategy]

    def summarize_conversation_turn(
        self,
        turn: ConversationTurn,
        strategy: Optional[SummaryStrategy] = None,
        max_length: Optional[int] = None
    ) -> SummaryResult:
        """
        摘要单个对话轮次

        Args:
            turn: 对话轮次对象
            strategy: 摘要策略
            max_length: 最大长度

        Returns:
            SummaryResult: 摘要结果
        """
        strategy = strategy or self.default_strategy
        summarizer = self._get_summarizer(strategy)

        # 合并问题和回答
        text = f"问题：{turn.question}\n回答：{turn.answer}"

        return summarizer.summarize(
            text,
            max_length=max_length,
            ratio=self.default_ratio
        )

    def summarize_session(
        self,
        session: ConversationSession,
        strategy: Optional[SummaryStrategy] = None,
        max_length: Optional[int] = None,
        include_qa: bool = True
    ) -> SummaryResult:
        """
        摘要整个对话会话

        Args:
            session: 对话会话对象
            strategy: 摘要策略
            max_length: 最大长度
            include_qa: 是否包含问答格式标记

        Returns:
            SummaryResult: 摘要结果
        """
        strategy = strategy or self.default_strategy
        summarizer = self._get_summarizer(strategy)

        # 构建完整对话文本
        conversation_text = []
        for i, turn in enumerate(session.turns, 1):
            if include_qa:
                turn_text = f"[第{i}轮]\n用户：{turn.question}\n助手：{turn.answer}"
            else:
                turn_text = f"{turn.question} {turn.answer}"
            conversation_text.append(turn_text)

        full_text = "\n\n".join(conversation_text)

        return summarizer.summarize(
            full_text,
            max_length=max_length,
            ratio=self.default_ratio
        )

    def summarize_multiple_sessions(
        self,
        sessions: List[ConversationSession],
        strategy: Optional[SummaryStrategy] = None,
        max_length: Optional[int] = None
    ) -> SummaryResult:
        """
        摘要多个对话会话

        Args:
            sessions: 对话会话列表
            strategy: 摘要策略
            max_length: 最大长度

        Returns:
            SummaryResult: 摘要结果
        """
        strategy = strategy or self.default_strategy
        summarizer = self._get_summarizer(strategy)

        # 合并多个会话
        merged_text = []
        for i, session in enumerate(sessions, 1):
            session_title = session.title or f"对话{i}"
            merged_text.append(f"=== {session_title} ===")

            for turn in session.turns:
                merged_text.append(f"Q: {turn.question}")
                merged_text.append(f"A: {turn.answer}")
            merged_text.append("")

        full_text = "\n".join(merged_text)

        return summarizer.summarize(
            full_text,
            max_length=max_length,
            ratio=self.default_ratio
        )

    def generate_topic_summary(
        self,
        text: str,
        strategy: Optional[SummaryStrategy] = None
    ) -> Dict[str, Any]:
        """
        生成带主题分析的摘要

        Args:
            text: 输入文本
            strategy: 摘要策略

        Returns:
            Dict: 包含主题、摘要、关键词等信息的字典
        """
        strategy = strategy or self.default_strategy
        summarizer = self._get_summarizer(strategy)

        result = summarizer.summarize(text, ratio=self.default_ratio)

        # 主题分段（简单实现：按段落分组）
        segments = self._segment_by_topic(text)

        return {
            "topic": result.topic,
            "summary": result.summary,
            "keywords": result.keywords,
            "key_points": result.key_points,
            "topic_segments": [
                {
                    "topic": seg.topic,
                    "summary": "".join(seg.sentences)[:200],
                    "keywords": seg.keywords
                } for seg in segments
            ],
            "statistics": {
                "original_length": result.original_length,
                "summary_length": result.summary_length,
                "compression_ratio": result.compression_ratio,
                "strategy": result.strategy
            }
        }

    def _segment_by_topic(self, text: str, min_segment_size: int = 3) -> List[TopicSegment]:
        """
        按主题分段（简单实现）

        Args:
            text: 输入文本
            min_segment_size: 最小段句子数

        Returns:
            List[TopicSegment]: 主题段落列表
        """
        sentences = TextPreprocessor.split_sentences(text)
        if len(sentences) < min_segment_size:
            keywords = TextPreprocessor.extract_keywords(text)
            return [TopicSegment(
                start_idx=0,
                end_idx=len(sentences),
                topic=keywords[0] if keywords else "全文",
                sentences=sentences,
                keywords=keywords
            )]

        # 简单分段：基于相似度聚类
        segments = []
        current_segment = [sentences[0]]
        current_start = 0

        for i in range(1, len(sentences)):
            # 计算与当前段的相似度
            segment_text = "".join(current_segment)
            sim = TextPreprocessor.calculate_similarity(segment_text, sentences[i])

            if sim < 0.1 and len(current_segment) >= min_segment_size:
                # 相似度低，新建分段
                seg_text = "".join(current_segment)
                keywords = TextPreprocessor.extract_keywords(seg_text)
                segments.append(TopicSegment(
                    start_idx=current_start,
                    end_idx=i,
                    topic=keywords[0] if keywords else f"段落{len(segments) + 1}",
                    sentences=current_segment.copy(),
                    keywords=keywords
                ))
                current_segment = [sentences[i]]
                current_start = i
            else:
                current_segment.append(sentences[i])

        # 处理最后一段
        if current_segment:
            seg_text = "".join(current_segment)
            keywords = TextPreprocessor.extract_keywords(seg_text)
            segments.append(TopicSegment(
                start_idx=current_start,
                end_idx=len(sentences),
                topic=keywords[0] if keywords else f"段落{len(segments) + 1}",
                sentences=current_segment,
                keywords=keywords
            ))

        return segments

    def summarize_document(
        self,
        text: str,
        strategy: Optional[SummaryStrategy] = None,
        length: SummaryLength = SummaryLength.MEDIUM,
        max_length: Optional[int] = None
    ) -> SummaryResult:
        """
        通用文档摘要函数

        Args:
            text: 文档文本
            strategy: 摘要策略
            length: 摘要长度枚举
            max_length: 可选的最大长度

        Returns:
            SummaryResult: 摘要结果
        """
        strategy = strategy or self.default_strategy
        summarizer = self._get_summarizer(strategy)

        # 如果指定了max_length则优先使用，否则使用length枚举
        if max_length is None:
            # 根据length枚举值计算目标长度
            char_count = len(text)
            max_length = int(char_count * length.value)

        return summarizer.summarize(
            text,
            max_length=max_length,
            ratio=length.value
        )

    def batch_summarize(
        self,
        documents: List[str],
        strategy: Optional[SummaryStrategy] = None,
        length: SummaryLength = SummaryLength.MEDIUM
    ) -> List[SummaryResult]:
        """
        批量摘要多个文档

        Args:
            documents: 文档列表
            strategy: 摘要策略
            length: 摘要长度

        Returns:
            List[SummaryResult]: 摘要结果列表
        """
        results = []
        for doc in documents:
            result = self.summarize_document(doc, strategy, length)
            results.append(result)
        return results

    def get_conversation_insights(
        self,
        session: ConversationSession
    ) -> Dict[str, Any]:
        """
        获取对话深度分析洞察

        Args:
            session: 对话会话

        Returns:
            Dict: 分析洞察结果
        """
        if not session.turns:
            return {
                "total_turns": 0,
                "message": "无对话数据可分析"
            }

        # 基础统计
        stats = session.get_statistics()

        # 所有问题和回答文本
        all_questions = [turn.question for turn in session.turns]
        all_answers = [turn.answer for turn in session.turns]
        all_text = " ".join(all_questions + all_answers)

        # 关键词分析
        keywords = TextPreprocessor.extract_keywords(all_text, top_n=15)

        # 主题分布
        topic_distribution = Counter(turn.category.value for turn in session.turns)

        # 质量趋势
        quality_trend = [
            {
                "turn": i + 1,
                "score": turn.quality_score,
                "level": turn.quality_level.value
            } for i, turn in enumerate(session.turns)
        ]

        # 平均响应时间趋势
        response_time_trend = [turn.response_time_ms for turn in session.turns]

        # 代码问题比例
        code_query_count = sum(1 for turn in session.turns if turn.is_code_query)
        code_query_ratio = code_query_count / len(session.turns) * 100

        # 生成会话摘要
        summary_result = self.summarize_session(session)

        return {
            "basic_stats": stats,
            "keywords": keywords,
            "topic_distribution": dict(topic_distribution),
            "quality_trend": quality_trend,
            "response_time_trend_ms": response_time_trend,
            "code_query_analysis": {
                "count": code_query_count,
                "ratio_percent": round(code_query_ratio, 1),
                "is_code_heavy": code_query_ratio > 50
            },
            "summary": summary_result.summary,
            "key_points": summary_result.key_points,
            "conversation_title": session.title,
            "duration_minutes": stats.get("duration_minutes", 0)
        }


# 全局分析器实例
_analyzer_instance: Optional[ConversationAnalyzer] = None


def get_conversation_analyzer(
    strategy: SummaryStrategy = SummaryStrategy.HYBRID
) -> ConversationAnalyzer:
    """
    获取对话分析器单例实例

    Args:
        strategy: 默认摘要策略

    Returns:
        ConversationAnalyzer: 分析器实例
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ConversationAnalyzer(default_strategy=strategy)
    return _analyzer_instance
