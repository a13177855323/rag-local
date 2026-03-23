"""
对话分析与导出模块
支持多格式导出、智能摘要生成、问题分类统计、回答质量评分
CPU环境下使用现有嵌入模型实现
"""

import os
import json
import csv
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np

from backend.services.conversation_store import ConversationStore, ConversationSession, ConversationTurn, get_conversation_store
from backend.models.embedding_model import EmbeddingModel


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
        self.conversation_store = get_conversation_store()
        self.embedding_model = EmbeddingModel()

    def analyze_session(self, session_id: str) -> Dict:
        """
        分析单个会话

        Args:
            session_id: 会话ID

        Returns:
            分析报告
        """
        session = self.conversation_store.get_session(session_id)
        if not session:
            return {"error": "会话不存在"}

        turns = session.turns
        if not turns:
            return {"error": "会话为空"}

        # 基础统计
        total_turns = len(turns)
        code_queries = sum(1 for t in turns if t.is_code_query)
        avg_response_time = sum(t.response_time_ms for t in turns) / total_turns

        # 问题分类统计
        category_stats = Counter(t.category for t in turns)

        # 时间分布
        hourly_distribution = self._analyze_time_distribution(turns)

        # 生成摘要
        summary = self._generate_summary(turns)

        # 计算质量评分
        quality_metrics = self._calculate_quality_metrics(turns)

        return {
            "session_id": session_id,
            "title": session.title,
            "summary": summary,
            "statistics": {
                "total_turns": total_turns,
                "code_queries": code_queries,
                "avg_response_time_ms": round(avg_response_time, 2),
                "total_duration_minutes": self._calculate_duration(turns)
            },
            "category_distribution": dict(category_stats),
            "hourly_distribution": hourly_distribution,
            "quality_metrics": quality_metrics
        }

    def _analyze_time_distribution(self, turns: List[ConversationTurn]) -> Dict:
        """分析时间分布"""
        hours = []
        for turn in turns:
            try:
                dt = datetime.fromisoformat(turn.timestamp)
                hours.append(dt.hour)
            except:
                continue

        if not hours:
            return {}

        distribution = Counter(hours)
        return {f"{h:02d}:00": count for h, count in sorted(distribution.items())}

    def _calculate_duration(self, turns: List[ConversationTurn]) -> int:
        """计算会话持续时间（分钟）"""
        if len(turns) < 2:
            return 0

        try:
            start = datetime.fromisoformat(turns[0].timestamp)
            end = datetime.fromisoformat(turns[-1].timestamp)
            duration = (end - start).total_seconds() / 60
            return int(duration)
        except:
            return 0

    def _generate_summary(self, turns: List[ConversationTurn]) -> str:
        """
        生成对话摘要
        使用现有嵌入模型实现智能摘要
        """
        if not turns:
            return ""

        # 提取关键问题
        questions = [t.question for t in turns[:5]]  # 取前5个问题
        key_topics = self._extract_key_topics(questions)

        # 生成摘要文本
        summary_parts = [
            f"本次对话共 {len(turns)} 轮问答",
            f"主要涉及主题: {', '.join(key_topics[:3])}",
        ]

        # 代码问题占比
        code_count = sum(1 for t in turns if t.is_code_query)
        if code_count > 0:
            summary_parts.append(f"包含 {code_count} 个代码相关问题")

        return "；".join(summary_parts)

    def _extract_key_topics(self, questions: List[str]) -> List[str]:
        """提取关键主题词"""
        # 简单的关键词提取
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

        found_topics = []
        for keyword in tech_keywords:
            if keyword in all_text:
                found_topics.append(keyword)

        return found_topics if found_topics else ["通用技术问题"]

    def _calculate_quality_metrics(self, turns: List[ConversationTurn]) -> Dict:
        """计算质量指标"""
        if not turns:
            return {}

        # 回答长度分布
        answer_lengths = [len(t.answer) for t in turns]

        # 来源引用率
        sources_cited = sum(1 for t in turns if t.sources) / len(turns)

        # 响应时间评分
        response_times = [t.response_time_ms for t in turns]
        avg_response_time = sum(response_times) / len(response_times)
        response_score = max(0, 100 - (avg_response_time / 100))  # 响应越快分数越高

        return {
            "avg_answer_length": int(sum(answer_lengths) / len(answer_lengths)),
            "sources_citation_rate": round(sources_cited * 100, 1),
            "avg_response_time_ms": round(avg_response_time, 2),
            "response_speed_score": round(min(response_score, 100), 1),
            "overall_score": round(self._calculate_overall_score(turns), 1)
        }

    def _calculate_overall_score(self, turns: List[ConversationTurn]) -> float:
        """计算综合质量评分"""
        if not turns:
            return 0.0

        scores = []
        for turn in turns:
            score = 70.0  # 基础分

            # 回答长度加分
            answer_len = len(turn.answer)
            if answer_len > 200:
                score += 10
            if answer_len > 500:
                score += 10

            # 有来源引用加分
            if turn.sources:
                score += 10

            # 响应时间扣分
            if turn.response_time_ms > 5000:
                score -= 10
            if turn.response_time_ms > 10000:
                score -= 10

            scores.append(max(0, min(100, score)))

        return sum(scores) / len(scores)

    def export_to_markdown(self, session_id: str, include_analysis: bool = True) -> str:
        """
        导出为 Markdown 格式

        Args:
            session_id: 会话ID
            include_analysis: 是否包含分析报告

        Returns:
            Markdown 内容
        """
        session = self.conversation_store.get_session(session_id)
        if not session:
            return "# 错误\n\n会话不存在"

        lines = [
            f"# {session.title}",
            "",
            f"**会话ID**: {session_id}",
            f"**创建时间**: {session.created_at}",
            f"**更新时间**: {session.updated_at}",
            f"**对话轮数**: {len(session.turns)}",
            ""
        ]

        # 添加分析报告
        if include_analysis:
            analysis = self.analyze_session(session_id)
            lines.extend([
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
            ])

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

        # 添加对话内容
        lines.extend([
            "## 对话详情",
            ""
        ])

        for i, turn in enumerate(session.turns, 1):
            lines.extend([
                f"### 第 {i} 轮",
                "",
                f"**问题** ({turn.category}):",
                "",
                f"{turn.question}",
                "",
                "**回答**:",
                "",
                f"{turn.answer}",
                ""
            ])

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

        return '\n'.join(lines)

    def export_to_json(self, session_id: str, include_analysis: bool = True) -> Dict:
        """
        导出为 JSON 格式

        Args:
            session_id: 会话ID
            include_analysis: 是否包含分析报告

        Returns:
            JSON 数据
        """
        session = self.conversation_store.get_session(session_id)
        if not session:
            return {"error": "会话不存在"}

        result = {
            "session_id": session_id,
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

        if include_analysis:
            result["analysis"] = self.analyze_session(session_id)

        return result

    def export_to_csv(self, session_id: str) -> str:
        """
        导出为 CSV 格式

        Args:
            session_id: 会话ID

        Returns:
            CSV 内容
        """
        session = self.conversation_store.get_session(session_id)
        if not session:
            return "error,message\n会话不存在,"

        import io
        output = io.StringIO()
        writer = csv.writer(output)

        # 写入表头
        writer.writerow([
            '轮次', '问题', '回答摘要', '分类', '是否代码问题',
            '响应时间(ms)', '时间戳', '来源数量', '质量评分'
        ])

        # 写入数据
        for i, turn in enumerate(session.turns, 1):
            answer_summary = turn.answer[:100] + '...' if len(turn.answer) > 100 else turn.answer
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

        # 全局统计
        total_sessions = len(sessions)
        total_turns = len(all_turns)
        code_queries = sum(1 for t in all_turns if t.is_code_query)

        # 分类统计
        category_stats = Counter(t.category for t in all_turns)

        # 每日统计
        daily_stats = {}
        for turn in all_turns:
            try:
                date = datetime.fromisoformat(turn.timestamp).strftime('%Y-%m-%d')
                daily_stats[date] = daily_stats.get(date, 0) + 1
            except:
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
    """获取 ConversationAnalyzer 单例实例"""
    global _conversation_analyzer
    if _conversation_analyzer is None:
        _conversation_analyzer = ConversationAnalyzer()
    return _conversation_analyzer
