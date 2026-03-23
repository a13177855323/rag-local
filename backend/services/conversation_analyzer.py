"""
对话分析与导出模块
支持多格式导出、智能摘要生成、问题分类统计、回答质量评分
CPU环境下使用现有嵌入模型实现

重构改进:
- 模块化设计: 分析逻辑拆分为独立子函数
- 配置管理: 使用 ConversationConfig 集中管理配置
- 异常处理: 完善的异常捕获和日志记录
- PEP8 规范: 统一的命名和代码风格
"""

import os
import csv
import io
import re
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter

from backend.services.conversation_store import (
    ConversationStore, ConversationSession, ConversationTurn, get_conversation_store
)
from backend.models.embedding_model import EmbeddingModel
from backend.utils.logger import get_logger, log_info, log_error, log_warning
from backend.utils.conversation_config import get_config


# 模块常量
MODULE_NAME = "ConversationAnalyzer"


class QualityCalculator:
    """质量评分计算器"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()

    def calculate_metrics(self, turns: List[ConversationTurn]) -> Dict:
        """
        计算质量指标

        Args:
            turns: 对话轮次列表

        Returns:
            质量指标字典
        """
        if not turns:
            return {}

        try:
            # 回答长度分布
            answer_lengths = [len(t.answer) for t in turns]

            # 来源引用率
            sources_cited = sum(1 for t in turns if t.sources) / len(turns)

            # 响应时间评分
            response_times = [t.response_time_ms for t in turns]
            avg_response_time = sum(response_times) / len(response_times)
            response_score = max(0, 100 - (avg_response_time / 100))

            metrics = {
                "avg_answer_length": int(sum(answer_lengths) / len(answer_lengths)),
                "sources_citation_rate": round(sources_cited * 100, 1),
                "avg_response_time_ms": round(avg_response_time, 2),
                "response_speed_score": round(min(response_score, 100), 1),
                "overall_score": round(self._calculate_overall_score(turns), 1)
            }

            log_info(
                MODULE_NAME,
                "Quality metrics calculated",
                {"turn_count": len(turns), "overall_score": metrics["overall_score"]}
            )

            return metrics

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Failed to calculate quality metrics",
                {"error": str(e)}
            )
            return {}

    def _calculate_overall_score(self, turns: List[ConversationTurn]) -> float:
        """
        计算综合质量评分

        Args:
            turns: 对话轮次列表

        Returns:
            综合评分 (0-100)
        """
        if not turns:
            return 0.0

        scores = []
        cfg = self.config

        for turn in turns:
            score = cfg.quality_base_score

            # 回答长度加分
            answer_len = len(turn.answer)
            if answer_len > cfg.quality_length_threshold_1:
                score += cfg.quality_length_bonus_1
            if answer_len > cfg.quality_length_threshold_2:
                score += cfg.quality_length_bonus_2

            # 有来源引用加分
            if turn.sources:
                score += cfg.quality_sources_bonus

            # 响应时间扣分
            if turn.response_time_ms > cfg.quality_slow_response_threshold:
                score -= cfg.quality_slow_penalty
            if turn.response_time_ms > cfg.quality_very_slow_response_threshold:
                score -= cfg.quality_slow_penalty

            scores.append(max(0, min(100, score)))

        return sum(scores) / len(scores)


class SummaryGenerator:
    """对话摘要生成器"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()

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

        try:
            # 提取关键问题
            questions = [t.question for t in turns[:self.config.summary_max_questions]]
            key_topics = self._extract_key_topics(questions)

            # 生成摘要文本
            summary_parts = [
                f"本次对话共 {len(turns)} 轮问答",
                f"主要涉及主题: {', '.join(key_topics[:self.config.summary_max_topics])}",
            ]

            # 代码问题占比
            code_count = sum(1 for t in turns if t.is_code_query)
            if code_count > 0:
                summary_parts.append(f"包含 {code_count} 个代码相关问题")

            summary = "；".join(summary_parts)

            log_info(
                MODULE_NAME,
                "Summary generated",
                {"turn_count": len(turns), "topic_count": len(key_topics)}
            )

            return summary

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Failed to generate summary",
                {"error": str(e)}
            )
            return f"对话共 {len(turns)} 轮问答"

    def _extract_key_topics(self, questions: List[str]) -> List[str]:
        """
        提取关键主题词

        Args:
            questions: 问题列表

        Returns:
            主题词列表
        """
        all_text = " ".join(questions).lower()

        # 技术关键词库
        tech_keywords = [
            "python", "java", "javascript", "react", "vue", "angular",
            "docker", "kubernetes", "aws", "azure", "gcp",
            "machine learning", "deep learning", "ai", "llm",
            "database", "sql", "nosql", "mongodb", "redis",
            "api", "rest", "graphql", "microservices",
            "git", "ci/cd", "devops", "linux"
        ]

        found_topics = [kw for kw in tech_keywords if kw in all_text]

        return found_topics if found_topics else ["通用技术问题"]


class TimeAnalyzer:
    """时间分布分析器"""

    def analyze_hourly_distribution(self, turns: List[ConversationTurn]) -> Dict:
        """
        分析时间分布

        Args:
            turns: 对话轮次列表

        Returns:
            小时分布字典
        """
        hours = []

        for turn in turns:
            try:
                dt = datetime.fromisoformat(turn.timestamp)
                hours.append(dt.hour)
            except (ValueError, TypeError) as e:
                log_warning(
                    MODULE_NAME,
                    "Invalid timestamp format",
                    {"timestamp": turn.timestamp, "error": str(e)}
                )
                continue

        if not hours:
            return {}

        distribution = Counter(hours)
        return {f"{h:02d}:00": count for h, count in sorted(distribution.items())}

    def calculate_duration(self, turns: List[ConversationTurn]) -> int:
        """
        计算会话持续时间（分钟）

        Args:
            turns: 对话轮次列表

        Returns:
            持续时间（分钟）
        """
        if len(turns) < 2:
            return 0

        try:
            start = datetime.fromisoformat(turns[0].timestamp)
            end = datetime.fromisoformat(turns[-1].timestamp)
            duration = (end - start).total_seconds() / 60
            return int(duration)

        except (ValueError, TypeError) as e:
            log_warning(
                MODULE_NAME,
                "Failed to calculate duration",
                {"error": str(e)}
            )
            return 0


class ExportFormatter:
    """导出格式处理器"""

    def __init__(self):
        self.config = get_config()

    def format_markdown(
        self,
        session: ConversationSession,
        analysis: Optional[Dict]
    ) -> str:
        """
        格式化为 Markdown

        Args:
            session: 会话对象
            analysis: 分析结果

        Returns:
            Markdown 内容
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

        # 添加分析报告
        if analysis:
            lines.extend(self._format_analysis_section(analysis))

        # 添加对话内容
        lines.extend([
            "## 对话详情",
            ""
        ])

        for i, turn in enumerate(session.turns, 1):
            lines.extend(self._format_turn_markdown(turn, i))

        return '\n'.join(lines)

    def _format_analysis_section(self, analysis: Dict) -> List[str]:
        """格式化分析部分"""
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
            "",
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

    def _format_turn_markdown(self, turn: ConversationTurn, index: int) -> List[str]:
        """格式化单轮对话为 Markdown"""
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
            for j, source in enumerate(turn.sources[:self.config.export_max_sources], 1):
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

    def format_json(
        self,
        session: ConversationSession,
        analysis: Optional[Dict]
    ) -> Dict:
        """
        格式化为 JSON

        Args:
            session: 会话对象
            analysis: 分析结果

        Returns:
            JSON 数据字典
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

        if analysis:
            result["analysis"] = analysis

        return result

    def format_csv(self, session: ConversationSession) -> str:
        """
        格式化为 CSV

        Args:
            session: 会话对象

        Returns:
            CSV 内容
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # 写入表头
        writer.writerow([
            '轮次', '问题', '回答摘要', '分类', '是否代码问题',
            '响应时间(ms)', '时间戳', '来源数量', '质量评分'
        ])

        # 写入数据
        for i, turn in enumerate(session.turns, 1):
            preview_len = self.config.export_answer_preview_length
            answer_summary = (
                turn.answer[:preview_len] + '...'
                if len(turn.answer) > preview_len
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
    """对话分析器 - 单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化分析器"""
        self.logger = get_logger()
        self.config = get_config()

        try:
            self.conversation_store = get_conversation_store()
            self.embedding_model = EmbeddingModel()

            # 初始化子组件
            self.quality_calculator = QualityCalculator()
            self.summary_generator = SummaryGenerator()
            self.time_analyzer = TimeAnalyzer()
            self.export_formatter = ExportFormatter()

            log_info(MODULE_NAME, "Analyzer initialized successfully")

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Failed to initialize analyzer",
                {"error": str(e)}
            )
            raise

    def analyze_session(self, session_id: str) -> Dict:
        """
        分析单个会话

        Args:
            session_id: 会话ID

        Returns:
            分析报告
        """
        try:
            session = self.conversation_store.get_session(session_id)

            if not session:
                log_warning(
                    MODULE_NAME,
                    "Session not found for analysis",
                    {"session_id": session_id}
                )
                return {"error": "会话不存在"}

            turns = session.turns
            if not turns:
                return {"error": "会话为空"}

            total_turns = len(turns)
            code_queries = sum(1 for t in turns if t.is_code_query)
            avg_response_time = sum(t.response_time_ms for t in turns) / total_turns

            result = {
                "session_id": session_id,
                "title": session.title,
                "summary": self.summary_generator.generate(turns),
                "statistics": {
                    "total_turns": total_turns,
                    "code_queries": code_queries,
                    "avg_response_time_ms": round(avg_response_time, 2),
                    "total_duration_minutes": self.time_analyzer.calculate_duration(turns)
                },
                "category_distribution": dict(Counter(t.category for t in turns)),
                "hourly_distribution": self.time_analyzer.analyze_hourly_distribution(turns),
                "quality_metrics": self.quality_calculator.calculate_metrics(turns)
            }

            log_info(
                MODULE_NAME,
                "Session analyzed",
                {"session_id": session_id, "turns": total_turns}
            )

            return result

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Failed to analyze session",
                {"session_id": session_id, "error": str(e)}
            )
            return {"error": f"分析失败: {str(e)}"}

    def export_to_markdown(self, session_id: str, include_analysis: bool = True) -> str:
        """
        导出为 Markdown 格式

        Args:
            session_id: 会话ID
            include_analysis: 是否包含分析报告

        Returns:
            Markdown 内容
        """
        try:
            session = self.conversation_store.get_session(session_id)

            if not session:
                log_warning(
                    MODULE_NAME,
                    "Session not found for export",
                    {"session_id": session_id, "format": "markdown"}
                )
                return "# 错误\n\n会话不存在"

            analysis = None
            if include_analysis:
                analysis = self.analyze_session(session_id)
                if "error" in analysis:
                    analysis = None

            content = self.export_formatter.format_markdown(session, analysis)

            log_info(
                MODULE_NAME,
                "Exported to markdown",
                {"session_id": session_id, "include_analysis": include_analysis}
            )

            return content

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Failed to export to markdown",
                {"session_id": session_id, "error": str(e)}
            )
            return f"# 错误\n\n导出失败: {str(e)}"

    def export_to_json(self, session_id: str, include_analysis: bool = True) -> Dict:
        """
        导出为 JSON 格式

        Args:
            session_id: 会话ID
            include_analysis: 是否包含分析报告

        Returns:
            JSON 数据
        """
        try:
            session = self.conversation_store.get_session(session_id)

            if not session:
                log_warning(
                    MODULE_NAME,
                    "Session not found for export",
                    {"session_id": session_id, "format": "json"}
                )
                return {"error": "会话不存在"}

            analysis = None
            if include_analysis:
                analysis = self.analyze_session(session_id)
                if "error" in analysis:
                    analysis = None

            result = self.export_formatter.format_json(session, analysis)

            log_info(
                MODULE_NAME,
                "Exported to JSON",
                {"session_id": session_id, "include_analysis": include_analysis}
            )

            return result

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Failed to export to JSON",
                {"session_id": session_id, "error": str(e)}
            )
            return {"error": f"导出失败: {str(e)}"}

    def export_to_csv(self, session_id: str) -> str:
        """
        导出为 CSV 格式

        Args:
            session_id: 会话ID

        Returns:
            CSV 内容
        """
        try:
            session = self.conversation_store.get_session(session_id)

            if not session:
                log_warning(
                    MODULE_NAME,
                    "Session not found for export",
                    {"session_id": session_id, "format": "csv"}
                )
                return "error,message\n会话不存在,"

            content = self.export_formatter.format_csv(session)

            log_info(
                MODULE_NAME,
                "Exported to CSV",
                {"session_id": session_id, "rows": len(session.turns)}
            )

            return content

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Failed to export to CSV",
                {"session_id": session_id, "error": str(e)}
            )
            return f"error,message\n导出失败,{str(e)}"

    def get_global_statistics(self) -> Dict:
        """
        获取全局统计信息

        Returns:
            全局统计数据
        """
        try:
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

            # 每日统计
            daily_stats = {}
            for turn in all_turns:
                try:
                    date = datetime.fromisoformat(turn.timestamp).strftime('%Y-%m-%d')
                    daily_stats[date] = daily_stats.get(date, 0) + 1
                except (ValueError, TypeError):
                    continue

            result = {
                "total_sessions": total_sessions,
                "total_turns": total_turns,
                "code_query_ratio": round(code_queries / total_turns * 100, 1),
                "category_distribution": dict(Counter(t.category for t in all_turns)),
                "daily_distribution": dict(sorted(daily_stats.items())),
                "avg_turns_per_session": round(total_turns / total_sessions, 1)
            }

            log_info(
                MODULE_NAME,
                "Global statistics calculated",
                {"sessions": total_sessions, "turns": total_turns}
            )

            return result

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Failed to calculate global statistics",
                {"error": str(e)}
            )
            return {"error": f"统计失败: {str(e)}"}


# 全局单例实例
_conversation_analyzer = None


def get_conversation_analyzer() -> ConversationAnalyzer:
    """获取 ConversationAnalyzer 单例实例"""
    global _conversation_analyzer
    if _conversation_analyzer is None:
        _conversation_analyzer = ConversationAnalyzer()
    return _conversation_analyzer
