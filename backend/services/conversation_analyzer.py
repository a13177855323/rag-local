"""
对话分析与导出模块

支持多格式导出、智能摘要生成、问题分类统计、回答质量评分。
CPU环境下使用现有嵌入模型实现智能摘要生成。

遵循PEP8编码规范，包含完整的异常处理和日志记录。

Author: RAG System
Date: 2026-03-23
"""

import csv
import io
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.config import settings
from backend.models.embedding_model import EmbeddingModel
from backend.services.conversation_store import (
    ConversationSession,
    ConversationStore,
    ConversationTurn,
    get_conversation_store,
)


# 配置日志格式 - 统一格式：时间戳 - 错误等级 - [模块标识] - 详细描述
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyzerError(Exception):
    """分析器相关异常基类"""
    pass


class TokenizerLoadError(AnalyzerError):
    """Tokenizer加载失败异常"""
    pass


class ModelWeightError(AnalyzerError):
    """模型权重文件损坏异常"""
    pass


class ExportError(AnalyzerError):
    """导出失败异常"""
    pass


class AnalysisError(AnalyzerError):
    """分析失败异常"""
    pass


class AnalyzerConfig:
    """
    分析器配置管理类

    管理所有分析相关的配置参数，支持CPU优化参数配置。
    """

    DEFAULT_CONFIG = {
        'max_summary_questions': 5,
        'max_key_topics': 3,
        'quality_base_score': 70.0,
        'quality_length_bonus_200': 10.0,
        'quality_length_bonus_500': 10.0,
        'quality_source_bonus': 10.0,
        'quality_response_penalty_5s': 10.0,
        'quality_response_penalty_10s': 10.0,
        'response_speed_baseline_ms': 100.0,
        'csv_answer_max_length': 100,
        'tech_keywords': [
            'python', 'java', 'javascript', 'react', 'vue', 'angular',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'machine learning', 'deep learning', 'ai', 'llm',
            'database', 'sql', 'nosql', 'mongodb', 'redis',
            'api', 'rest', 'graphql', 'microservices',
            'git', 'ci/cd', 'devops', 'linux'
        ]
    }

    def __init__(self, custom_config: Optional[Dict] = None):
        """
        初始化配置

        Args:
            custom_config: 自定义配置字典，会覆盖默认配置
        """
        self._config = self.DEFAULT_CONFIG.copy()
        if custom_config:
            self._config.update(custom_config)
        logger.info("分析器配置初始化完成")

    def get(self, key: str, default=None):
        """获取配置项"""
        return self._config.get(key, default)

    def update(self, key: str, value) -> None:
        """更新配置项"""
        self._config[key] = value

    @property
    def tech_keywords(self) -> List[str]:
        """获取技术关键词列表"""
        return self._config['tech_keywords']


class TokenizerManager:
    """
    Tokenizer管理器 - 独立处理tokenizer相关操作

    负责tokenizer的加载、初始化和文本处理。
    包含完整的异常处理机制。
    """

    def __init__(self, embedding_model: EmbeddingModel):
        """
        初始化Tokenizer管理器

        Args:
            embedding_model: 嵌入模型实例

        Raises:
            TokenizerLoadError: tokenizer加载失败时抛出
            ModelWeightError: 模型权重损坏时抛出
        """
        self.embedding_model = embedding_model
        self._tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self) -> None:
        """
        初始化tokenizer

        包含tokenizer加载失败的捕获与处理机制。
        """
        try:
            if hasattr(self.embedding_model, 'tokenizer'):
                self._tokenizer = self.embedding_model.tokenizer
                logger.info("Tokenizer初始化成功: 使用嵌入模型的tokenizer")
            else:
                self._tokenizer = None
                logger.info("Tokenizer初始化: 使用简化分词模式")
        except FileNotFoundError as e:
            logger.error(f"Tokenizer文件不存在: {e}")
            raise TokenizerLoadError(f"Tokenizer文件不存在: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Tokenizer配置文件格式错误: {e}")
            raise TokenizerLoadError(f"Tokenizer配置文件格式错误: {e}")
        except Exception as e:
            logger.warning(f"Tokenizer初始化异常，使用简化模式: {e}")
            self._tokenizer = None

    def tokenize(self, text: str) -> List[str]:
        """
        分词处理

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        if not text:
            return []

        if self._tokenizer is not None:
            try:
                tokens = self._tokenizer.tokenize(text)
                return tokens if tokens else self._simple_tokenize(text)
            except Exception as e:
                logger.warning(f"Tokenizer分词失败，使用简化模式: {e}")
                return self._simple_tokenize(text)

        return self._simple_tokenize(text)

    def _simple_tokenize(self, text: str) -> List[str]:
        """
        简化分词方法 - 作为tokenizer不可用时的后备方案

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        tokens = text.split()
        return [t.lower() for t in tokens if len(t) > 1]

    def count_tokens(self, text: str) -> int:
        """
        计算token数量

        Args:
            text: 输入文本

        Returns:
            token数量
        """
        return len(self.tokenize(text))


class EmbeddingProcessor:
    """
    嵌入向量处理器 - 独立处理嵌入相关操作

    负责文本嵌入生成和相似度计算。
    """

    def __init__(self, embedding_model: EmbeddingModel):
        """
        初始化嵌入处理器

        Args:
            embedding_model: 嵌入模型实例

        Raises:
            ModelWeightError: 模型权重损坏时抛出
        """
        self.embedding_model = embedding_model
        self._validate_model()

    def _validate_model(self) -> None:
        """
        验证模型权重完整性

        Raises:
            ModelWeightError: 模型权重损坏时抛出
        """
        try:
            test_embedding = self.embedding_model.embed_query("test")
            if test_embedding is None or len(test_embedding) == 0:
                raise ModelWeightError("模型权重验证失败: 无法生成嵌入向量")
            logger.info("嵌入模型权重验证通过")
        except Exception as e:
            logger.error(f"模型权重验证异常: {e}")
            raise ModelWeightError(f"模型权重验证失败: {e}")

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量，失败返回None
        """
        try:
            embedding = self.embedding_model.embed_query(text)
            return np.array(embedding) if embedding else None
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return None

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        计算两个嵌入向量的余弦相似度

        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量

        Returns:
            相似度分数 (0-1)
        """
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(np.clip(similarity, 0.0, 1.0))
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0


class QualityScorer:
    """
    质量评分器 - 独立处理质量评分逻辑

    实现回答质量的综合评分计算。
    """

    def __init__(self, config: AnalyzerConfig):
        """
        初始化质量评分器

        Args:
            config: 分析器配置实例
        """
        self.config = config

    def calculate_turn_score(self, turn: ConversationTurn) -> float:
        """
        计算单轮对话的质量评分

        Args:
            turn: 对话轮次对象

        Returns:
            质量评分 (0-100)
        """
        score = self.config.get('quality_base_score')

        answer_len = len(turn.answer)
        if answer_len > 200:
            score += self.config.get('quality_length_bonus_200')
        if answer_len > 500:
            score += self.config.get('quality_length_bonus_500')

        if turn.sources:
            score += self.config.get('quality_source_bonus')

        if turn.response_time_ms > 5000:
            score -= self.config.get('quality_response_penalty_5s')
        if turn.response_time_ms > 10000:
            score -= self.config.get('quality_response_penalty_10s')

        return max(0.0, min(100.0, score))

    def calculate_session_score(self, turns: List[ConversationTurn]) -> float:
        """
        计算会话整体质量评分

        Args:
            turns: 对话轮次列表

        Returns:
            平均质量评分
        """
        if not turns:
            return 0.0

        scores = [self.calculate_turn_score(turn) for turn in turns]
        return sum(scores) / len(scores)


class SummaryGenerator:
    """
    摘要生成器 - 独立处理摘要生成逻辑

    使用现有嵌入模型实现智能摘要生成。
    """

    def __init__(
        self,
        config: AnalyzerConfig,
        embedding_processor: EmbeddingProcessor,
        tokenizer_manager: TokenizerManager
    ):
        """
        初始化摘要生成器

        Args:
            config: 分析器配置实例
            embedding_processor: 嵌入处理器实例
            tokenizer_manager: Tokenizer管理器实例
        """
        self.config = config
        self.embedding_processor = embedding_processor
        self.tokenizer_manager = tokenizer_manager

    def generate(self, turns: List[ConversationTurn]) -> str:
        """
        生成对话摘要

        Args:
            turns: 对话轮次列表

        Returns:
            摘要文本
        """
        if not turns:
            return ""

        max_questions = self.config.get('max_summary_questions')
        questions = [t.question for t in turns[:max_questions]]
        key_topics = self._extract_key_topics(questions)

        summary_parts = [
            f"本次对话共 {len(turns)} 轮问答",
            f"主要涉及主题: {', '.join(key_topics[:self.config.get('max_key_topics')])}"
        ]

        code_count = sum(1 for t in turns if t.is_code_query)
        if code_count > 0:
            summary_parts.append(f"包含 {code_count} 个代码相关问题")

        return "；".join(summary_parts)

    def _extract_key_topics(self, questions: List[str]) -> List[str]:
        """
        提取关键主题词

        Args:
            questions: 问题列表

        Returns:
            关键主题词列表
        """
        all_text = " ".join(questions).lower()
        tech_keywords = self.config.tech_keywords

        found_topics = []
        for keyword in tech_keywords:
            if keyword in all_text:
                found_topics.append(keyword)

        return found_topics if found_topics else ["通用技术问题"]


class ExportManager:
    """
    导出管理器 - 独立处理多格式导出

    支持Markdown、JSON、CSV格式的导出。
    """

    def __init__(self, config: AnalyzerConfig):
        """
        初始化导出管理器

        Args:
            config: 分析器配置实例
        """
        self.config = config

    def export_to_markdown(
        self,
        session: ConversationSession,
        analysis: Optional[Dict] = None,
        include_analysis: bool = True
    ) -> str:
        """
        导出为Markdown格式

        Args:
            session: 会话对象
            analysis: 分析结果
            include_analysis: 是否包含分析报告

        Returns:
            Markdown内容
        """
        lines = [
            f"# {session.title}",
            "",
            f"**会话ID**: {session.id}",
            f"**创建时间**: {session.created_at}",
            f"**更新时间**: {session.updated_at}",
            f"**对话轮数**: {len(session.turns)}",
            ""
        ]

        if include_analysis and analysis:
            lines.extend(self._build_analysis_section(analysis))

        lines.extend([
            "## 对话详情",
            ""
        ])

        for i, turn in enumerate(session.turns, 1):
            lines.extend(self._build_turn_section(turn, i))

        return '\n'.join(lines)

    def _build_analysis_section(self, analysis: Dict) -> List[str]:
        """构建分析报告部分"""
        lines = [
            "## 对话分析",
            "",
            f"**摘要**: {analysis.get('summary', 'N/A')}",
            "",
            "### 统计信息",
            "",
            f"- 总轮数: {analysis['statistics']['total_turns']}",
            f"- 代码问题: {analysis['statistics']['code_queries']}",
            f"- 平均响应时间: {analysis['statistics']['avg_response_time_ms']}ms",
            f"- 持续时长: {analysis['statistics']['total_duration_minutes']}分钟",
            "",
            "### 问题分类",
            ""
        ]

        for category, count in analysis.get('category_distribution', {}).items():
            lines.append(f"- {category}: {count}")

        lines.extend([
            "",
            "### 质量指标",
            "",
            f"- 平均回答长度: {analysis['quality_metrics']['avg_answer_length']} 字符",
            f"- 来源引用率: {analysis['quality_metrics']['sources_citation_rate']}%",
            f"- 响应速度评分: {analysis['quality_metrics']['response_speed_score']}",
            f"- 综合评分: {analysis['quality_metrics']['overall_score']}",
            ""
        ])

        return lines

    def _build_turn_section(self, turn: ConversationTurn, index: int) -> List[str]:
        """构建对话轮次部分"""
        lines = [
            f"### 第 {index} 轮",
            "",
            f"**问题** ({turn.category}):",
            "",
            f"{turn.question}",
            "",
            "**回答**:",
            "",
            f"{turn.answer}",
            ""
        ]

        if turn.sources:
            lines.extend([
                "**参考来源**:",
                ""
            ])
            for j, source in enumerate(turn.sources[:3], 1):
                filename = source.get('filename', 'Unknown')
                similarity = source.get('similarity', 0)
                lines.append(f"{j}. {filename} (相似度: {similarity:.3f})")
            lines.append("")

        lines.extend([
            f"*时间: {turn.timestamp} | 响应: {turn.response_time_ms}ms*",
            "",
            "---",
            ""
        ])

        return lines

    def export_to_json(
        self,
        session: ConversationSession,
        analysis: Optional[Dict] = None,
        include_analysis: bool = True
    ) -> Dict:
        """
        导出为JSON格式

        Args:
            session: 会话对象
            analysis: 分析结果
            include_analysis: 是否包含分析报告

        Returns:
            JSON数据字典
        """
        result = {
            "session_id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "turns": [
                {
                    "id": turn.id,
                    "question": turn.question,
                    "answer": turn.answer,
                    "category": turn.category,
                    "is_code_query": turn.is_code_query,
                    "timestamp": turn.timestamp,
                    "response_time_ms": turn.response_time_ms,
                    "sources": turn.sources,
                    "quality_score": turn.quality_score
                }
                for turn in session.turns
            ]
        }

        if include_analysis and analysis:
            result["analysis"] = analysis

        return result

    def export_to_csv(self, session: ConversationSession) -> str:
        """
        导出为CSV格式

        Args:
            session: 会话对象

        Returns:
            CSV内容
        """
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            '轮次', '问题', '回答摘要', '分类', '是否代码问题',
            '响应时间(ms)', '时间戳', '来源数量', '质量评分'
        ])

        max_length = self.config.get('csv_answer_max_length')
        for i, turn in enumerate(session.turns, 1):
            answer_summary = (
                turn.answer[:max_length] + '...'
                if len(turn.answer) > max_length
                else turn.answer
            )
            writer.writerow([
                i,
                turn.question,
                answer_summary,
                turn.category,
                '是' if turn.is_code_query else '否',
                turn.response_time_ms,
                turn.timestamp,
                len(turn.sources),
                turn.quality_score
            ])

        return output.getvalue()


class ConversationAnalyzer:
    """
    对话分析器 - 主分析类

    整合各子模块，提供完整的对话分析功能。
    单例模式实现。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """初始化分析器组件"""
        self.conversation_store = get_conversation_store()
        self.config = AnalyzerConfig()
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        初始化各组件

        包含模型权重文件损坏的检测与异常捕获。
        """
        try:
            self.embedding_model = EmbeddingModel()
            self.tokenizer_manager = TokenizerManager(self.embedding_model)
            self.embedding_processor = EmbeddingProcessor(self.embedding_model)
            self.quality_scorer = QualityScorer(self.config)
            self.summary_generator = SummaryGenerator(
                self.config,
                self.embedding_processor,
                self.tokenizer_manager
            )
            self.export_manager = ExportManager(self.config)
            logger.info("分析器组件初始化完成")
        except ModelWeightError as e:
            logger.error(f"模型权重加载失败: {e}")
            raise
        except Exception as e:
            logger.error(f"分析器组件初始化异常: {e}")
            raise AnalyzerError(f"分析器初始化失败: {e}")

    def analyze_session(self, session_id: str) -> Dict:
        """
        分析单个会话

        Args:
            session_id: 会话ID

        Returns:
            分析报告字典
        """
        session = self.conversation_store.get_session(session_id)
        if not session:
            return {"error": "会话不存在"}

        turns = session.turns
        if not turns:
            return {"error": "会话为空"}

        try:
            statistics = self._calculate_statistics(turns)
            category_distribution = self._analyze_category_distribution(turns)
            hourly_distribution = self._analyze_time_distribution(turns)
            quality_metrics = self._calculate_quality_metrics(turns)
            summary = self.summary_generator.generate(turns)

            return {
                "session_id": session_id,
                "title": session.title,
                "summary": summary,
                "statistics": statistics,
                "category_distribution": category_distribution,
                "hourly_distribution": hourly_distribution,
                "quality_metrics": quality_metrics
            }
        except Exception as e:
            logger.error(f"分析会话失败: {e}")
            return {"error": f"分析失败: {e}"}

    def _calculate_statistics(self, turns: List[ConversationTurn]) -> Dict:
        """计算基础统计信息"""
        total_turns = len(turns)
        code_queries = sum(1 for t in turns if t.is_code_query)
        avg_response_time = sum(t.response_time_ms for t in turns) / total_turns
        duration = self._calculate_duration(turns)

        return {
            "total_turns": total_turns,
            "code_queries": code_queries,
            "avg_response_time_ms": round(avg_response_time, 2),
            "total_duration_minutes": duration
        }

    def _calculate_duration(self, turns: List[ConversationTurn]) -> int:
        """计算会话持续时间（分钟）"""
        if len(turns) < 2:
            return 0

        try:
            start = datetime.fromisoformat(turns[0].timestamp)
            end = datetime.fromisoformat(turns[-1].timestamp)
            duration = (end - start).total_seconds() / 60
            return int(duration)
        except Exception:
            return 0

    def _analyze_category_distribution(self, turns: List[ConversationTurn]) -> Dict:
        """分析问题分类分布"""
        distribution = Counter(t.category for t in turns)
        return dict(distribution)

    def _analyze_time_distribution(self, turns: List[ConversationTurn]) -> Dict:
        """分析时间分布"""
        hours = []
        for turn in turns:
            try:
                dt = datetime.fromisoformat(turn.timestamp)
                hours.append(dt.hour)
            except Exception:
                continue

        if not hours:
            return {}

        distribution = Counter(hours)
        return {f"{h:02d}:00": count for h, count in sorted(distribution.items())}

    def _calculate_quality_metrics(self, turns: List[ConversationTurn]) -> Dict:
        """计算质量指标"""
        if not turns:
            return {}

        answer_lengths = [len(t.answer) for t in turns]
        sources_cited = sum(1 for t in turns if t.sources) / len(turns)
        response_times = [t.response_time_ms for t in turns]
        avg_response_time = sum(response_times) / len(response_times)

        baseline = self.config.get('response_speed_baseline_ms')
        response_score = max(0, 100 - (avg_response_time / baseline))

        return {
            "avg_answer_length": int(sum(answer_lengths) / len(answer_lengths)),
            "sources_citation_rate": round(sources_cited * 100, 1),
            "avg_response_time_ms": round(avg_response_time, 2),
            "response_speed_score": round(min(response_score, 100), 1),
            "overall_score": round(self.quality_scorer.calculate_session_score(turns), 1)
        }

    def export_to_markdown(
        self,
        session_id: str,
        include_analysis: bool = True
    ) -> str:
        """
        导出为Markdown格式

        Args:
            session_id: 会话ID
            include_analysis: 是否包含分析报告

        Returns:
            Markdown内容
        """
        session = self.conversation_store.get_session(session_id)
        if not session:
            return "# 错误\n\n会话不存在"

        analysis = None
        if include_analysis:
            analysis = self.analyze_session(session_id)

        return self.export_manager.export_to_markdown(session, analysis, include_analysis)

    def export_to_json(
        self,
        session_id: str,
        include_analysis: bool = True
    ) -> Dict:
        """
        导出为JSON格式

        Args:
            session_id: 会话ID
            include_analysis: 是否包含分析报告

        Returns:
            JSON数据字典
        """
        session = self.conversation_store.get_session(session_id)
        if not session:
            return {"error": "会话不存在"}

        analysis = None
        if include_analysis:
            analysis = self.analyze_session(session_id)

        return self.export_manager.export_to_json(session, analysis, include_analysis)

    def export_to_csv(self, session_id: str) -> str:
        """
        导出为CSV格式

        Args:
            session_id: 会话ID

        Returns:
            CSV内容
        """
        session = self.conversation_store.get_session(session_id)
        if not session:
            return "error,message\n会话不存在,"

        return self.export_manager.export_to_csv(session)

    def get_global_statistics(self) -> Dict:
        """获取全局统计信息"""
        sessions = self.conversation_store.get_all_sessions()
        all_turns = []

        for session_data in sessions:
            session = self.conversation_store.get_session(session_data['id'])
            if session:
                all_turns.extend(session.turns)

        if not all_turns:
            return {"message": "暂无对话数据"}

        total_sessions = len(sessions)
        total_turns = len(all_turns)
        code_queries = sum(1 for t in all_turns if t.is_code_query)

        category_stats = Counter(t.category for t in all_turns)

        daily_stats = {}
        for turn in all_turns:
            try:
                date = datetime.fromisoformat(turn.timestamp).strftime('%Y-%m-%d')
                daily_stats[date] = daily_stats.get(date, 0) + 1
            except Exception:
                continue

        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "code_query_ratio": round(code_queries / total_turns * 100, 1),
            "category_distribution": dict(category_stats),
            "daily_distribution": dict(sorted(daily_stats.items())),
            "avg_turns_per_session": round(total_turns / total_sessions, 1)
        }


# 全局单例实例
_conversation_analyzer = None


def get_conversation_analyzer() -> ConversationAnalyzer:
    """
    获取 ConversationAnalyzer 单例实例

    Returns:
        ConversationAnalyzer 实例
    """
    global _conversation_analyzer
    if _conversation_analyzer is None:
        _conversation_analyzer = ConversationAnalyzer()
    return _conversation_analyzer
