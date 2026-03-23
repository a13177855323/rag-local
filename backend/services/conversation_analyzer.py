import re
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass
from backend.services.conversation_history import ConversationTurn, ConversationSession
from backend.models.embedding_model import EmbeddingModel


@dataclass
class ConversationAnalysis:
    summary: str
    total_turns: int
    category_stats: Dict[str, int]
    avg_quality_score: float
    avg_response_time_ms: float
    topics: List[str]
    quality_distribution: Dict[str, int]


class QuestionClassifier:
    CATEGORIES = {
        "概念解释": [
            r"什么是", r"解释", r"定义", r"含义", r"意思",
            r"介绍", r"说明", r"概念", r"理解"
        ],
        "操作指导": [
            r"如何", r"怎么", r"怎样", r"方法", r"步骤",
            r"操作", r"实现", r"配置", r"设置", r"使用"
        ],
        "问题排查": [
            r"错误", r"异常", r"失败", r"报错", r"问题",
            r"为什么", r"原因", r"解决", r"修复", r"调试"
        ],
        "代码相关": [
            r"代码", r"函数", r"类", r"方法", r"变量",
            r"实现", r"编写", r"示例", r"python", r"api"
        ],
        "对比分析": [
            r"区别", r"对比", r"比较", r"差异", r"优劣",
            r"选择", r"哪个", r"更好"
        ],
        "其他": []
    }

    @classmethod
    def classify(cls, question: str) -> str:
        question_lower = question.lower()
        for category, patterns in cls.CATEGORIES.items():
            if category == "其他":
                continue
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return category
        return "其他"


class QualityScorer:
    @staticmethod
    def score(
        question: str,
        answer: str,
        sources: List[Dict],
        response_time_ms: float = None
    ) -> float:
        score = 0.0

        answer_length = len(answer)
        if answer_length > 50:
            score += 0.2
        if answer_length > 200:
            score += 0.1
        if answer_length > 500:
            score += 0.1

        if sources and len(sources) > 0:
            score += min(len(sources) * 0.1, 0.3)

        avg_similarity = 0.0
        if sources:
            similarities = [s.get("similarity", 0) for s in sources]
            avg_similarity = sum(similarities) / len(similarities)
            score += min(avg_similarity * 0.2, 0.2)

        if response_time_ms:
            if response_time_ms < 1000:
                score += 0.1
            elif response_time_ms < 3000:
                score += 0.05

        question_words = set(question.split())
        answer_words = set(answer.split())
        overlap = len(question_words & answer_words)
        if overlap > 0:
            score += min(overlap * 0.02, 0.1)

        return min(score, 1.0)


class ConversationAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.embedding_model = EmbeddingModel()
        self.classifier = QuestionClassifier()
        self.scorer = QualityScorer()

    def classify_question(self, question: str) -> str:
        return self.classifier.classify(question)

    def score_turn(self, turn: ConversationTurn) -> float:
        return self.scorer.score(
            turn.question,
            turn.answer,
            turn.sources,
            turn.response_time_ms
        )

    def analyze_session(self, session: ConversationSession) -> ConversationAnalysis:
        if not session.turns:
            return ConversationAnalysis(
                summary="无对话记录",
                total_turns=0,
                category_stats={},
                avg_quality_score=0.0,
                avg_response_time_ms=0.0,
                topics=[],
                quality_distribution={}
            )

        categories = []
        quality_scores = []
        response_times = []
        all_questions = []

        for turn in session.turns:
            category = self.classify_question(turn.question)
            categories.append(category)

            score = self.score_turn(turn)
            quality_scores.append(score)

            if turn.response_time_ms:
                response_times.append(turn.response_time_ms)

            all_questions.append(turn.question)

        category_stats = dict(Counter(categories))

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        quality_distribution = {
            "优秀": sum(1 for s in quality_scores if s >= 0.8),
            "良好": sum(1 for s in quality_scores if 0.6 <= s < 0.8),
            "一般": sum(1 for s in quality_scores if 0.4 <= s < 0.6),
            "较差": sum(1 for s in quality_scores if s < 0.4)
        }

        topics = self._extract_topics(all_questions)

        summary = self._generate_summary(session, category_stats, avg_quality)

        return ConversationAnalysis(
            summary=summary,
            total_turns=len(session.turns),
            category_stats=category_stats,
            avg_quality_score=round(avg_quality, 3),
            avg_response_time_ms=round(avg_response_time, 2),
            topics=topics,
            quality_distribution=quality_distribution
        )

    def _extract_topics(self, questions: List[str]) -> List[str]:
        if not questions:
            return []

        all_text = " ".join(questions)

        keywords = re.findall(r'[\u4e00-\u9fa5]{2,4}|[a-zA-Z]{3,}', all_text)

        stop_words = {"什么", "怎么", "如何", "为什么", "可以", "能够", "这个", "那个",
                     "一个", "是否", "有没有", "能不能", "会不会", "的话", "现在"}

        filtered = [w for w in keywords if w.lower() not in stop_words]

        word_freq = Counter(filtered)
        top_topics = [word for word, _ in word_freq.most_common(5)]

        return top_topics

    def _generate_summary(
        self,
        session: ConversationSession,
        category_stats: Dict[str, int],
        avg_quality: float
    ) -> str:
        parts = []

        parts.append(f"本次对话共 {len(session.turns)} 轮问答。")

        if category_stats:
            top_category = max(category_stats, key=category_stats.get)
            parts.append(f"主要问题类型为「{top_category}」({category_stats[top_category]}次)。")

        if avg_quality >= 0.8:
            quality_desc = "优秀"
        elif avg_quality >= 0.6:
            quality_desc = "良好"
        elif avg_quality >= 0.4:
            quality_desc = "一般"
        else:
            quality_desc = "有待提升"
        parts.append(f"整体回答质量{quality_desc}(评分{avg_quality:.2f})。")

        if session.turns:
            first_q = session.turns[0].question[:30]
            last_q = session.turns[-1].question[:30]
            parts.append(f"首问: {first_q}...")
            parts.append(f"末问: {last_q}...")

        return " ".join(parts)

    def analyze_multiple_sessions(
        self,
        sessions: List[ConversationSession]
    ) -> Dict:
        if not sessions:
            return {
                "total_sessions": 0,
                "total_turns": 0,
                "overall_category_stats": {},
                "overall_avg_quality": 0.0,
                "daily_stats": []
            }

        all_categories = []
        all_quality_scores = []
        daily_stats = []
        total_turns = 0

        for session in sessions:
            analysis = self.analyze_session(session)
            all_categories.extend(list(analysis.category_stats.keys()) * len(session.turns))
            all_quality_scores.append(analysis.avg_quality_score)
            total_turns += analysis.total_turns

            daily_stats.append({
                "date": session.created_at[:10],
                "turns": analysis.total_turns,
                "avg_quality": analysis.avg_quality_score,
                "top_category": max(analysis.category_stats, key=analysis.category_stats.get) if analysis.category_stats else "无"
            })

        overall_category_stats = dict(Counter(all_categories))
        overall_avg_quality = sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0

        return {
            "total_sessions": len(sessions),
            "total_turns": total_turns,
            "overall_category_stats": overall_category_stats,
            "overall_avg_quality": round(overall_avg_quality, 3),
            "daily_stats": sorted(daily_stats, key=lambda x: x["date"], reverse=True)
        }


conversation_analyzer = ConversationAnalyzer()


def get_analyzer() -> ConversationAnalyzer:
    return conversation_analyzer
