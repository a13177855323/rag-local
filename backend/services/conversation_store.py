"""
对话历史存储模块
支持对话记录的存储、查询、导出和分析
采用类型安全的数据结构和高效的持久化存储机制
"""

import os
import json
import uuid
import shutil
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime
from threading import Lock
from collections import defaultdict, Counter
import re


class QuestionCategory(Enum):
    """问题分类枚举"""
    CODE = "代码问题"
    CONCEPT = "概念解释"
    HOW_TO = "操作指南"
    DEBUG = "调试排错"
    COMPARISON = "对比分析"
    OTHER = "其他"

    @classmethod
    def from_string(cls, category_str: str) -> 'QuestionCategory':
        """从字符串转换为枚举值"""
        mapping = {
            "代码问题": cls.CODE,
            "概念解释": cls.CONCEPT,
            "操作指南": cls.HOW_TO,
            "调试排错": cls.DEBUG,
            "对比分析": cls.COMPARISON,
            "其他": cls.OTHER
        }
        return mapping.get(category_str, cls.OTHER)


class QualityLevel(Enum):
    """回答质量等级"""
    EXCELLENT = "优秀"
    GOOD = "良好"
    FAIR = "一般"
    POOR = "较差"
    UNKNOWN = "未知"


@dataclass
class SourceReference:
    """来源引用数据结构"""
    filename: str
    content: str
    similarity: float = 0.0
    chunk_id: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceReference':
        """从字典创建实例"""
        return cls(
            filename=data.get('filename', ''),
            content=data.get('content', ''),
            similarity=data.get('similarity', 0.0),
            chunk_id=data.get('chunk_id', 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ConversationTurn:
    """
    单轮对话数据结构

    存储一轮完整的问答对，包括问题、回答、来源引用、元数据等。
    """
    id: str
    session_id: str
    question: str
    answer: str
    sources: List[SourceReference] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response_time_ms: int = 0
    is_code_query: bool = False
    category: QuestionCategory = QuestionCategory.OTHER
    quality_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.UNKNOWN
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """从字典创建实例"""
        # 处理来源引用
        sources = []
        for src in data.get('sources', []):
            if isinstance(src, dict):
                sources.append(SourceReference.from_dict(src))
            elif isinstance(src, SourceReference):
                sources.append(src)

        # 处理分类
        category = data.get('category', '其他')
        if isinstance(category, str):
            category = QuestionCategory.from_string(category)

        # 处理质量等级
        quality_level = data.get('quality_level', '未知')
        if isinstance(quality_level, str):
            quality_level = QualityLevel(quality_level) if quality_level in [x.value for x in QualityLevel] else QualityLevel.UNKNOWN

        return cls(
            id=data.get('id', str(uuid.uuid4())),
            session_id=data.get('session_id', ''),
            question=data.get('question', ''),
            answer=data.get('answer', ''),
            sources=sources,
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            response_time_ms=data.get('response_time_ms', 0),
            is_code_query=data.get('is_code_query', False),
            category=category,
            quality_score=data.get('quality_score', 0.0),
            quality_level=quality_level,
            feedback=data.get('feedback'),
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        # 转换枚举值为字符串
        result['category'] = self.category.value if isinstance(self.category, QuestionCategory) else str(self.category)
        result['quality_level'] = self.quality_level.value if isinstance(self.quality_level, QualityLevel) else str(self.quality_level)
        # 转换来源引用
        result['sources'] = [src.to_dict() if isinstance(src, SourceReference) else src for src in self.sources]
        return result

    def get_word_count(self) -> Tuple[int, int]:
        """获取问题和回答的字数"""
        question_words = len(re.findall(r'\w+|\S', self.question))
        answer_words = len(re.findall(r'\w+|\S', self.answer))
        return question_words, answer_words

    def calculate_quality_score(self) -> float:
        """
        计算回答质量分数

        评分维度：
        1. 回答长度（是否足够详细）
        2. 来源引用数量（可信度）
        3. 响应时间（用户体验）
        4. 代码相关性（如果是代码问题）

        Returns:
            float: 质量分数，范围 [0, 100]
        """
        score = 50.0  # 基础分

        # 1. 回答长度评分
        answer_length = len(self.answer)
        if answer_length > 500:
            score += 20
        elif answer_length > 200:
            score += 15
        elif answer_length > 100:
            score += 10
        elif answer_length < 50:
            score -= 10

        # 2. 来源引用评分
        source_count = len(self.sources)
        if source_count >= 3:
            score += 15
        elif source_count >= 2:
            score += 10
        elif source_count >= 1:
            score += 5

        # 3. 响应时间评分（毫秒）
        if self.response_time_ms < 2000:
            score += 10
        elif self.response_time_ms < 5000:
            score += 5
        elif self.response_time_ms > 15000:
            score -= 10

        # 4. 代码问题特殊评分
        if self.is_code_query:
            # 检查回答中是否包含代码块（使用```标记）
            if '```' in self.answer:
                score += 10
            elif '`' in self.answer or '代码' in self.answer:
                score += 5

        # 确保分数在0-100范围内
        self.quality_score = max(0.0, min(100.0, score))

        # 设置质量等级
        if self.quality_score >= 80:
            self.quality_level = QualityLevel.EXCELLENT
        elif self.quality_score >= 65:
            self.quality_level = QualityLevel.GOOD
        elif self.quality_score >= 50:
            self.quality_level = QualityLevel.FAIR
        else:
            self.quality_level = QualityLevel.POOR

        return self.quality_score


@dataclass
class ConversationSession:
    """
    对话会话数据结构

    存储一个完整的对话会话，包含多个问答轮次和会话元数据。
    """
    id: str
    title: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    turns: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """从字典创建实例"""
        turns = []
        for turn_data in data.get('turns', []):
            turns.append(ConversationTurn.from_dict(turn_data))

        return cls(
            id=data.get('id', str(uuid.uuid4())),
            title=data.get('title', 'Untitled'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            turns=turns,
            metadata=data.get('metadata', {}),
            tags=data.get('tags', [])
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['turns'] = [turn.to_dict() for turn in self.turns]
        return result

    def add_turn(self, turn: ConversationTurn) -> None:
        """添加对话轮次"""
        self.turns.append(turn)
        self.updated_at = datetime.now().isoformat()

        # 如果是第一轮对话，自动设置标题（使用问题的前50字符）
        if len(self.turns) == 1 and self.title == 'Untitled':
            self.title = turn.question[:50] + ("..." if len(turn.question) > 50 else "")

    def get_turn_by_id(self, turn_id: str) -> Optional[ConversationTurn]:
        """根据ID获取对话轮次"""
        for turn in self.turns:
            if turn.id == turn_id:
                return turn
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        if not self.turns:
            return {}

        total_turns = len(self.turns)
        code_queries = sum(1 for t in self.turns if t.is_code_query)
        avg_response_time = sum(t.response_time_ms for t in self.turns) / total_turns
        avg_quality_score = sum(t.quality_score for t in self.turns) / total_turns

        # 分类统计
        category_stats = Counter(t.category.value for t in self.turns)

        # 总字数
        total_question_words = sum(t.get_word_count()[0] for t in self.turns)
        total_answer_words = sum(t.get_word_count()[1] for t in self.turns)

        # 持续时间
        try:
            start = datetime.fromisoformat(self.created_at)
            end = datetime.fromisoformat(self.updated_at)
            duration_minutes = (end - start).total_seconds() / 60
        except:
            duration_minutes = 0

        return {
            "total_turns": total_turns,
            "code_queries": code_queries,
            "code_query_ratio": round(code_queries / total_turns * 100, 1),
            "avg_response_time_ms": round(avg_response_time, 2),
            "avg_quality_score": round(avg_quality_score, 2),
            "category_distribution": dict(category_stats),
            "total_question_words": total_question_words,
            "total_answer_words": total_answer_words,
            "duration_minutes": round(duration_minutes, 1)
        }


class ConversationStore:
    """
    对话存储管理器 - 单例模式

    提供对话数据的持久化存储、查询和管理功能，支持：
    1. 会话的创建、查询、更新和删除
    2. 对话轮次的添加和修改
    3. 高效的JSON持久化存储
    4. 线程安全操作
    5. 数据备份和恢复

    Attributes:
        storage_dir: 存储目录路径
        sessions_file: 会话数据文件路径
        sessions: 内存中的会话字典
        lock: 线程锁
    """

    _instance: Optional['ConversationStore'] = None
    _initialized: bool = False

    def __new__(cls) -> 'ConversationStore':
        """创建或获取单例实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """初始化存储管理器"""
        if self._initialized:
            return

        self._initialize()
        self._initialized = True

    def _initialize(self) -> None:
        """初始化存储"""
        from backend.config import settings

        self.storage_dir = os.path.join("./data", "conversations")
        os.makedirs(self.storage_dir, exist_ok=True)

        self.sessions_file = os.path.join(self.storage_dir, "sessions.json")
        self.backup_dir = os.path.join(self.storage_dir, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)

        # 内存中的会话存储：session_id -> ConversationSession
        self.sessions: Dict[str, ConversationSession] = {}

        # 线程锁
        self.lock = Lock()

        # 分类关键词映射（用于自动分类）
        self._setup_category_keywords()

        # 从磁盘加载数据
        self._load_sessions()

        print(f"对话存储初始化完成，已加载 {len(self.sessions)} 个会话")

    def _setup_category_keywords(self) -> None:
        """设置分类关键词映射"""
        # 按优先级顺序定义分类关键词（优先级高的在前）
        self.category_keywords = {
            QuestionCategory.HOW_TO: [
                '如何', '怎么', 'how to', '怎样', '如何实现', '怎么弄', '步骤',
                '教程', 'guide', '使用方法', '操作指南', '安装', '配置'
            ],
            QuestionCategory.DEBUG: [
                '报错', '错误', 'error', 'exception', '失败', '无法', '不能',
                '问题', 'bug', '调试', 'debug', '修复', 'fix'
            ],
            QuestionCategory.CODE: [
                '代码', 'code', '函数', 'function', '类', 'class', '方法', 'method',
                'python', 'java', 'javascript', 'js', 'cpp', 'c++', '脚本', '编程',
                '写一段', '实现一个', '代码示例'
            ],
            QuestionCategory.COMPARISON: [
                '区别', '对比', '比较', 'difference', 'compare', 'vs', 'versus',
                '哪个好', '优劣', '差异', '不同'
            ],
            QuestionCategory.CONCEPT: [
                '什么是', '概念', '原理', '机制', 'what is', 'explain', '解释',
                '定义', '含义', '理解'
            ]
        }

    def _classify_question(self, question: str) -> QuestionCategory:
        """
        对问题进行智能分类

        Args:
            question: 问题文本

        Returns:
            QuestionCategory: 问题分类
        """
        question_lower = question.lower()

        # 最高优先级：比较分析问题（包含编程语言和比较关键词）
        comparison_keywords = ['区别', '对比', '比较', 'difference', 'compare', 'vs', 'versus', '哪个好', '优劣', '差异', '不同']
        has_comparison = any(kw in question_lower for kw in comparison_keywords)

        if has_comparison:
            return QuestionCategory.COMPARISON

        # 特殊规则：如果同时包含编程语言和"问题"但没有明确的调试词汇，归类为代码问题
        code_languages = ['python', 'java', 'javascript', 'js', 'cpp', 'c++', 'c语言', 'golang', 'rust']
        has_language = any(lang in question_lower for lang in code_languages)
        has_code_keyword = any(kw in question_lower for kw in ['代码', '函数', '类', '方法'])
        has_debug_keyword = any(kw in question_lower for kw in ['报错', '错误', 'exception', 'bug', '调试', 'fix'])

        if (has_language or has_code_keyword) and not has_debug_keyword and ('问题' in question_lower or '怎么' in question_lower):
            return QuestionCategory.CODE

        # 按优先级顺序检查分类关键词
        for category, keywords in self.category_keywords.items():
            for kw in keywords:
                if kw in question_lower:
                    return category

        return QuestionCategory.OTHER

    def _load_sessions(self) -> None:
        """从磁盘加载会话数据"""
        if not os.path.exists(self.sessions_file):
            return

        try:
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for session_id, session_data in data.items():
                    try:
                        session = ConversationSession.from_dict(session_data)
                        self.sessions[session_id] = session
                    except Exception as e:
                        print(f"加载会话 {session_id} 失败: {e}")
        except Exception as e:
            print(f"加载会话数据失败: {e}，创建新的存储文件")
            self.sessions = {}

    def _save_sessions(self, create_backup: bool = True) -> None:
        """
        保存会话数据到磁盘

        Args:
            create_backup: 是否在保存前创建备份
        """
        with self.lock:
            try:
                # 创建备份
                if create_backup and os.path.exists(self.sessions_file):
                    self._create_backup()

                # 序列化会话数据
                data = {}
                for session_id, session in self.sessions.items():
                    data[session_id] = session.to_dict()

                # 保存到文件
                with open(self.sessions_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"保存会话数据失败: {e}")
                raise

    def _create_backup(self) -> None:
        """创建会话数据备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"sessions_{timestamp}.json")

        try:
            shutil.copy2(self.sessions_file, backup_file)

            # 只保留最近10个备份
            backups = sorted([f for f in os.listdir(self.backup_dir) if f.startswith('sessions_')])
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    os.remove(os.path.join(self.backup_dir, old_backup))
        except Exception as e:
            print(f"创建备份失败: {e}")

    def create_session(self, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        创建新的对话会话

        Args:
            title: 会话标题
            metadata: 会话元数据

        Returns:
            str: 新会话的ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        if not title:
            title = f"对话 {now[:19].replace('T', ' ')}"

        session = ConversationSession(
            id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            turns=[],
            metadata=metadata or {},
            tags=[]
        )

        with self.lock:
            self.sessions[session_id] = session

        self._save_sessions()
        return session_id

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict],
        response_time_ms: int = 0,
        is_code_query: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ConversationTurn]:
        """
        添加对话轮次

        Args:
            session_id: 会话ID
            question: 用户问题
            answer: 系统回答
            sources: 来源引用列表
            response_time_ms: 响应时间（毫秒）
            is_code_query: 是否为代码相关问题
            metadata: 附加元数据

        Returns:
            Optional[ConversationTurn]: 创建的对话轮次对象，失败返回None
        """
        # 如果会话不存在，创建新会话
        if session_id not in self.sessions:
            session_id = self.create_session(title=question[:50] + "..." if len(question) > 50 else question)

        # 转换来源引用
        source_refs = []
        for src in sources:
            if isinstance(src, dict):
                source_refs.append(SourceReference.from_dict(src))

        # 自动分类问题
        category = self._classify_question(question)

        # 创建对话轮次
        turn = ConversationTurn(
            id=str(uuid.uuid4()),
            session_id=session_id,
            question=question,
            answer=answer,
            sources=source_refs,
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time_ms,
            is_code_query=is_code_query,
            category=category,
            metadata=metadata or {}
        )

        # 计算质量分数
        turn.calculate_quality_score()

        # 添加到会话
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].add_turn(turn)

        self._save_sessions()
        return turn

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        获取指定会话

        Args:
            session_id: 会话ID

        Returns:
            Optional[ConversationSession]: 会话对象，不存在返回None
        """
        return self.sessions.get(session_id)

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        获取所有会话的摘要信息

        Returns:
            List[Dict]: 会话摘要列表，按更新时间排序
        """
        sessions_list = []
        for session in self.sessions.values():
            sessions_list.append({
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "turn_count": len(session.turns),
                "tags": session.tags,
                "metadata": session.metadata
            })

        # 按更新时间排序（最新的在前）
        sessions_list.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions_list

    def search_sessions(
        self,
        keyword: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        搜索包含关键词的会话

        Args:
            keyword: 搜索关键词
            limit: 返回结果数量限制

        Returns:
            List[Dict]: 匹配的会话摘要列表
        """
        keyword_lower = keyword.lower()
        results = []

        for session in self.sessions.values():
            # 搜索标题
            if keyword_lower in session.title.lower():
                results.append(session)
                continue

            # 搜索对话内容
            for turn in session.turns:
                if (keyword_lower in turn.question.lower() or
                        keyword_lower in turn.answer.lower()):
                    results.append(session)
                    break

            if len(results) >= limit:
                break

        # 转换为摘要格式
        return [{
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "turn_count": len(s.turns)
        } for s in results]

    def update_session_title(self, session_id: str, title: str) -> bool:
        """
        更新会话标题

        Args:
            session_id: 会话ID
            title: 新标题

        Returns:
            bool: 更新成功返回True，失败返回False
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        with self.lock:
            session.title = title
            session.updated_at = datetime.now().isoformat()

        self._save_sessions()
        return True

    def update_turn_feedback(self, session_id: str, turn_id: str, feedback: str) -> bool:
        """
        更新对话轮次的用户反馈

        Args:
            session_id: 会话ID
            turn_id: 轮次ID
            feedback: 用户反馈

        Returns:
            bool: 更新成功返回True，失败返回False
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        turn = session.get_turn_by_id(turn_id)
        if not turn:
            return False

        with self.lock:
            turn.feedback = feedback

        self._save_sessions()
        return True

    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话

        Args:
            session_id: 要删除的会话ID

        Returns:
            bool: 删除成功返回True，失败返回False
        """
        if session_id not in self.sessions:
            return False

        with self.lock:
            del self.sessions[session_id]

        self._save_sessions()
        return True

    def get_session_statistics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取指定会话的统计信息"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return session.get_statistics()

    def get_global_statistics(self) -> Dict[str, Any]:
        """
        获取全局对话统计信息

        Returns:
            Dict: 包含各类统计信息的字典
        """
        if not self.sessions:
            return {
                "total_sessions": 0,
                "total_turns": 0,
                "message": "暂无对话数据"
            }

        total_sessions = len(self.sessions)
        all_turns = []
        for session in self.sessions.values():
            all_turns.extend(session.turns)

        total_turns = len(all_turns)

        if total_turns == 0:
            return {
                "total_sessions": total_sessions,
                "total_turns": 0,
                "message": "暂无对话数据"
            }

        # 分类统计
        category_stats = Counter(t.category.value for t in all_turns)

        # 代码查询比例
        code_queries = sum(1 for t in all_turns if t.is_code_query)
        code_query_ratio = round(code_queries / total_turns * 100, 1)

        # 每日统计
        daily_stats = defaultdict(int)
        for turn in all_turns:
            try:
                date = datetime.fromisoformat(turn.timestamp).strftime('%Y-%m-%d')
                daily_stats[date] += 1
            except:
                continue

        # 质量分布
        quality_distribution = Counter(t.quality_level.value for t in all_turns)

        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "code_queries": code_queries,
            "code_query_ratio": code_query_ratio,
            "avg_response_time_ms": round(sum(t.response_time_ms for t in all_turns) / total_turns, 2),
            "avg_quality_score": round(sum(t.quality_score for t in all_turns) / total_turns, 2),
            "category_distribution": dict(category_stats),
            "quality_distribution": dict(quality_distribution),
            "daily_distribution": dict(sorted(daily_stats.items())),
            "avg_turns_per_session": round(total_turns / total_sessions, 1)
        }

    def export_session(
        self,
        session_id: str,
        format_type: str = 'json'
    ) -> Optional[Any]:
        """
        导出会话数据

        Args:
            session_id: 会话ID
            format_type: 导出格式 ('json', 'dict')

        Returns:
            Optional[Any]: 导出的数据，格式取决于format_type
        """
        session = self.sessions.get(session_id)
        if not session:
            return None

        if format_type == 'json':
            return json.dumps(session.to_dict(), ensure_ascii=False, indent=2)
        elif format_type == 'dict':
            return session.to_dict()
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")

    def clear_all_sessions(self) -> bool:
        """
        清除所有会话数据（危险操作！）

        Returns:
            bool: 清除成功返回True
        """
        try:
            # 先备份
            if self.sessions:
                self._create_backup()

            with self.lock:
                self.sessions.clear()

            self._save_sessions(create_backup=False)
            print("已清除所有会话数据")
            return True
        except Exception as e:
            print(f"清除会话数据失败: {e}")
            return False


# 全局单例实例
_conversation_store_instance: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """
    获取 ConversationStore 单例实例

    Returns:
        ConversationStore: 全局对话存储实例
    """
    global _conversation_store_instance
    if _conversation_store_instance is None:
        _conversation_store_instance = ConversationStore()
    return _conversation_store_instance
