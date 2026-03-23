"""
对话历史存储模块
支持对话记录的存储、查询、导出和分析
"""

import os
import json
import csv
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid


class QuestionCategory(Enum):
    """问题分类枚举"""
    CODE = "代码问题"
    CONCEPT = "概念解释"
    HOW_TO = "操作指南"
    DEBUG = "调试排错"
    COMPARISON = "对比分析"
    OTHER = "其他"


@dataclass
class ConversationTurn:
    """单轮对话数据结构"""
    id: str
    session_id: str
    question: str
    answer: str
    sources: List[Dict]
    timestamp: str
    response_time_ms: int
    is_code_query: bool = False
    category: str = "other"
    quality_score: float = 0.0


@dataclass
class ConversationSession:
    """对话会话数据结构"""
    id: str
    title: str
    created_at: str
    updated_at: str
    turns: List[ConversationTurn]
    metadata: Dict


class ConversationStore:
    """对话存储管理器 - 单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化存储"""
        self.storage_dir = os.path.join("./data", "conversations")
        os.makedirs(self.storage_dir, exist_ok=True)

        self.sessions_file = os.path.join(self.storage_dir, "sessions.json")
        self.sessions: Dict[str, ConversationSession] = {}

        self._load_sessions()

    def _load_sessions(self):
        """从磁盘加载会话数据"""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for session_id, session_data in data.items():
                        turns = [ConversationTurn(**turn) for turn in session_data.get('turns', [])]
                        self.sessions[session_id] = ConversationSession(
                            id=session_data['id'],
                            title=session_data['title'],
                            created_at=session_data['created_at'],
                            updated_at=session_data['updated_at'],
                            turns=turns,
                            metadata=session_data.get('metadata', {})
                        )
                print(f"加载了 {len(self.sessions)} 个对话会话")
            except Exception as e:
                print(f"加载会话数据失败: {e}")
                self.sessions = {}

    def _save_sessions(self):
        """保存会话数据到磁盘"""
        try:
            data = {}
            for session_id, session in self.sessions.items():
                data[session_id] = {
                    'id': session.id,
                    'title': session.title,
                    'created_at': session.created_at,
                    'updated_at': session.updated_at,
                    'turns': [asdict(turn) for turn in session.turns],
                    'metadata': session.metadata
                }
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存会话数据失败: {e}")

    def create_session(self, title: str = None) -> str:
        """
        创建新对话会话

        Args:
            title: 会话标题

        Returns:
            会话ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        if not title:
            title = f"对话 {now[:19]}"

        session = ConversationSession(
            id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            turns=[],
            metadata={}
        )

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
        is_code_query: bool = False
    ) -> ConversationTurn:
        """
        添加对话轮次

        Args:
            session_id: 会话ID
            question: 问题
            answer: 回答
            sources: 来源文档
            response_time_ms: 响应时间（毫秒）
            is_code_query: 是否为代码查询

        Returns:
            创建的对话轮次
        """
        if session_id not in self.sessions:
            session_id = self.create_session()

        turn = ConversationTurn(
            id=str(uuid.uuid4()),
            session_id=session_id,
            question=question,
            answer=answer,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time_ms,
            is_code_query=is_code_query,
            category=self._classify_question(question),
            quality_score=0.0  # 稍后计算
        )

        self.sessions[session_id].turns.append(turn)
        self.sessions[session_id].updated_at = datetime.now().isoformat()
        self._save_sessions()

        return turn

    def _classify_question(self, question: str) -> str:
        """
        对问题进行分类

        Args:
            question: 问题文本

        Returns:
            分类标签
        """
        question_lower = question.lower()

        # 代码相关问题
        code_keywords = ['代码', 'code', '函数', 'function', '类', 'class', '方法', 'method',
                        'python', 'java', 'javascript', 'js', 'cpp', 'c++', 'bug', '报错', 'error']
        if any(kw in question_lower for kw in code_keywords):
            return QuestionCategory.CODE.value

        # 操作指南
        how_to_keywords = ['如何', '怎么', 'how to', '步骤', '教程', 'guide', '步骤']
        if any(kw in question_lower for kw in how_to_keywords):
            return QuestionCategory.HOW_TO.value

        # 调试排错
        debug_keywords = ['报错', '错误', 'error', 'exception', '失败', '无法', '不能', 'debug']
        if any(kw in question_lower for kw in debug_keywords):
            return QuestionCategory.DEBUG.value

        # 对比分析
        compare_keywords = ['区别', '对比', '比较', 'difference', 'compare', 'vs', 'versus']
        if any(kw in question_lower for kw in compare_keywords):
            return QuestionCategory.COMPARISON.value

        # 概念解释
        concept_keywords = ['什么是', '什么是', '概念', '原理', '机制', '什么是', 'what is', 'explain']
        if any(kw in question_lower for kw in concept_keywords):
            return QuestionCategory.CONCEPT.value

        return QuestionCategory.OTHER.value

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取会话"""
        return self.sessions.get(session_id)

    def get_all_sessions(self) -> List[Dict]:
        """获取所有会话列表"""
        return [
            {
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "turn_count": len(session.turns)
            }
            for session in self.sessions.values()
        ]

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            return True
        return False

    def update_turn_quality_score(self, turn_id: str, score: float):
        """更新对话轮次的质量评分"""
        for session in self.sessions.values():
            for turn in session.turns:
                if turn.id == turn_id:
                    turn.quality_score = score
                    self._save_sessions()
                    return True
        return False


# 全局单例实例
_conversation_store = None


def get_conversation_store() -> ConversationStore:
    """获取 ConversationStore 单例实例"""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store
