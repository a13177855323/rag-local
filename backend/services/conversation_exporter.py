import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Optional
from io import StringIO, BytesIO
from backend.services.conversation_history import ConversationSession, ConversationTurn
from backend.services.conversation_analyzer import ConversationAnalysis, ConversationAnalyzer
from backend.config import settings


class ConversationExporter:
    def __init__(self):
        self.export_dir = os.path.join(settings.VECTOR_DB_PATH, "exports")
        os.makedirs(self.export_dir, exist_ok=True)

    def export_to_markdown(
        self,
        session: ConversationSession,
        analysis: Optional[ConversationAnalysis] = None,
        include_analysis: bool = True
    ) -> str:
        lines = []

        lines.append("# 对话历史报告")
        lines.append("")
        lines.append(f"**会话ID**: {session.session_id}")
        lines.append(f"**创建时间**: {session.created_at}")
        lines.append(f"**更新时间**: {session.updated_at}")
        lines.append(f"**对话轮数**: {len(session.turns)}")
        lines.append("")

        if include_analysis and analysis:
            lines.append("## 对话分析摘要")
            lines.append("")
            lines.append(f"**摘要**: {analysis.summary}")
            lines.append("")
            lines.append("### 问题分类统计")
            lines.append("")
            lines.append("| 类别 | 数量 |")
            lines.append("|------|------|")
            for category, count in analysis.category_stats.items():
                lines.append(f"| {category} | {count} |")
            lines.append("")

            lines.append("### 质量评估")
            lines.append("")
            lines.append(f"- **平均质量评分**: {analysis.avg_quality_score}")
            lines.append(f"- **平均响应时间**: {analysis.avg_response_time_ms}ms")
            lines.append("")
            lines.append("#### 质量分布")
            lines.append("")
            lines.append("| 等级 | 数量 |")
            lines.append("|------|------|")
            for level, count in analysis.quality_distribution.items():
                lines.append(f"| {level} | {count} |")
            lines.append("")

            if analysis.topics:
                lines.append(f"**主要话题**: {', '.join(analysis.topics)}")
                lines.append("")

        lines.append("## 对话详情")
        lines.append("")

        for i, turn in enumerate(session.turns, 1):
            lines.append(f"### 第 {i} 轮对话")
            lines.append("")
            lines.append(f"**时间**: {turn.timestamp}")
            if turn.question_category:
                lines.append(f"**问题类别**: {turn.question_category}")
            if turn.quality_score is not None:
                lines.append(f"**质量评分**: {turn.quality_score:.2f}")
            lines.append("")
            lines.append("**问题**:")
            lines.append("")
            lines.append(f"> {turn.question}")
            lines.append("")
            lines.append("**回答**:")
            lines.append("")
            lines.append(turn.answer)
            lines.append("")

            if turn.sources:
                lines.append("**参考来源**:")
                lines.append("")
                for j, source in enumerate(turn.sources, 1):
                    lines.append(f"{j}. {source.get('filename', '未知文件')} (相似度: {source.get('similarity', 0):.3f})")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def export_to_json(
        self,
        session: ConversationSession,
        analysis: Optional[ConversationAnalysis] = None,
        include_analysis: bool = True
    ) -> str:
        export_data = {
            "session": session.to_dict(),
            "export_time": datetime.now().isoformat()
        }

        if include_analysis and analysis:
            export_data["analysis"] = {
                "summary": analysis.summary,
                "total_turns": analysis.total_turns,
                "category_stats": analysis.category_stats,
                "avg_quality_score": analysis.avg_quality_score,
                "avg_response_time_ms": analysis.avg_response_time_ms,
                "topics": analysis.topics,
                "quality_distribution": analysis.quality_distribution
            }

        return json.dumps(export_data, ensure_ascii=False, indent=2)

    def export_to_csv(
        self,
        sessions: List[ConversationSession],
        include_analysis: bool = True
    ) -> str:
        output = StringIO()
        writer = csv.writer(output)

        headers = [
            "会话ID", "轮次", "时间", "问题", "回答",
            "问题类别", "质量评分", "响应时间(ms)", "来源数量"
        ]
        writer.writerow(headers)

        for session in sessions:
            for i, turn in enumerate(session.turns, 1):
                row = [
                    session.session_id,
                    i,
                    turn.timestamp,
                    turn.question[:100] + "..." if len(turn.question) > 100 else turn.question,
                    turn.answer[:100] + "..." if len(turn.answer) > 100 else turn.answer,
                    turn.question_category or "",
                    f"{turn.quality_score:.2f}" if turn.quality_score else "",
                    f"{turn.response_time_ms:.0f}" if turn.response_time_ms else "",
                    len(turn.sources)
                ]
                writer.writerow(row)

        return output.getvalue()

    def export_statistics_csv(
        self,
        sessions: List[ConversationSession],
        analyzer: ConversationAnalyzer
    ) -> str:
        output = StringIO()
        writer = csv.writer(output)

        headers = [
            "日期", "会话ID", "对话轮数", "平均质量评分",
            "主要问题类别", "平均响应时间(ms)"
        ]
        writer.writerow(headers)

        for session in sessions:
            analysis = analyzer.analyze_session(session)
            top_category = max(analysis.category_stats, key=analysis.category_stats.get) if analysis.category_stats else "无"

            row = [
                session.created_at[:10],
                session.session_id,
                analysis.total_turns,
                f"{analysis.avg_quality_score:.3f}",
                top_category,
                f"{analysis.avg_response_time_ms:.0f}"
            ]
            writer.writerow(row)

        return output.getvalue()

    def save_export(
        self,
        content: str,
        filename: str,
        format_type: str
    ) -> str:
        filepath = os.path.join(self.export_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def export_all_sessions(
        self,
        sessions: List[ConversationSession],
        analyzer: ConversationAnalyzer,
        format_type: str = "markdown"
    ) -> Dict[str, str]:
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type in ["markdown", "md"]:
            for session in sessions:
                analysis = analyzer.analyze_session(session)
                content = self.export_to_markdown(session, analysis)
                filename = f"conversation_{session.session_id}_{timestamp}.md"
                filepath = self.save_export(content, filename, "markdown")
                results[session.session_id] = filepath

        elif format_type == "json":
            all_data = {
                "export_time": datetime.now().isoformat(),
                "sessions": []
            }
            for session in sessions:
                analysis = analyzer.analyze_session(session)
                session_data = json.loads(self.export_to_json(session, analysis))
                all_data["sessions"].append(session_data)

            content = json.dumps(all_data, ensure_ascii=False, indent=2)
            filename = f"all_conversations_{timestamp}.json"
            filepath = self.save_export(content, filename, "json")
            results["all"] = filepath

        elif format_type == "csv":
            content = self.export_to_csv(sessions)
            filename = f"conversations_{timestamp}.csv"
            filepath = self.save_export(content, filename, "csv")
            results["conversations"] = filepath

            stats_content = self.export_statistics_csv(sessions, analyzer)
            stats_filename = f"conversation_stats_{timestamp}.csv"
            stats_filepath = self.save_export(stats_content, stats_filename, "csv")
            results["statistics"] = stats_filepath

        return results


conversation_exporter = ConversationExporter()


def get_exporter() -> ConversationExporter:
    return conversation_exporter
