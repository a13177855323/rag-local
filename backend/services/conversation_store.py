"""
对话历史存储模块

支持对话记录的存储、查询、导出和分析。
遵循PEP8编码规范，包含完整的异常处理和日志记录。

Author: RAG System
Date: 2026-03-23
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path

from backend.config import settings


# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class ConversationError(Exception):
    """对话存储相关异常基类"""
    pass


class SessionNotFoundError(ConversationError):
    """会话不存在异常"""
    pass


class SessionLoadError(ConversationError):
    """会话加载失败异常"""
    pass


class SessionSaveError(ConversationError):
    """会话保存失败异常"""
    pass


class InvalidSessionDataError(ConversationError):
    """无效会话数据异常"""
    pass


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
    """
    单轮对话数据结构

    Attributes:
        id: 对话轮次唯一标识
        session_id: 所属会话ID
        question: 用户问题
        answer: 系统回答
        sources: 参考来源列表
        timestamp: 时间戳
        response_time_ms: 响应时间(毫秒)
        is_code_query: 是否为代码查询
        category: 问题分类
        quality_score: 质量评分
    """
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

    def validate(self) -> bool:
        """验证数据完整性"""
        if not self.id or not self.session_id:
            return False
        if not isinstance(self.question, str) or not isinstance(self.answer, str):
            return False
        return True


@dataclass
class ConversationSession:
    """
    对话会话数据结构

    Attributes:
        id: 会话唯一标识
        title: 会话标题
        created_at: 创建时间
        updated_at: 更新时间
        turns: 对话轮次列表
        metadata: 元数据
    """
    id: str
    title: str
    created_at: str
    updated_at: str
    turns: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def validate(self) -> bool:
        """验证会话数据完整性"""
        if not self.id:
            return False
        if not self.title:
            return False
        return True


class QuestionClassifier:
    """问题分类器 - 支持可配置的关键词匹配"""

    CATEGORY_KEYWORDS = {
        QuestionCategory.CODE: [
            '代码', 'code', '函数', 'function', '类', 'class',
            '方法', 'method', 'python', 'java', 'javascript', 'js',
            'cpp', 'c++', 'bug', '报错', 'error', '实现'
        ],
        QuestionCategory.HOW_TO: [
            '如何', '怎么', 'how to', '步骤', '教程',
            'guide', '怎样', '方法'
        ],
        QuestionCategory.DEBUG: [
            '报错', '错误', 'error', 'exception', '失败',
            '无法', '不能', 'debug', '排查', '异常'
        ],
        QuestionCategory.COMPARISON: [
            '区别', '对比', '比较', 'difference', 'compare',
            'vs', 'versus', '差异', '优劣'
        ],
        QuestionCategory.CONCEPT: [
            '什么是', '概念', '原理', '机制', 'what is',
            'explain', '解释', '介绍'
        ],
    }

    @classmethod
    def classify(cls, question: str) -> str:
        """
        对问题进行分类

        Args:
            question: 问题文本

        Returns:
            分类标签字符串
        """
        if not question or not isinstance(question, str):
            return QuestionCategory.OTHER.value

        question_lower = question.lower()

        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(kw in question_lower for kw in keywords):
                return category.value

        return QuestionCategory.OTHER.value


class SessionFileManager:
    """会话文件管理器 - 处理文件读写操作"""

    def __init__(self, storage_dir: str):
        """
        初始化文件管理器

        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.sessions_file = self.storage_dir / "sessions.json"
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """确保存储目录存在"""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"存储目录已就绪: {self.storage_dir}")
        except OSError as e:
            logger.error(f"创建存储目录失败: {e}")
            raise SessionSaveError(f"无法创建存储目录: {e}")

    def load_sessions(self) -> Dict:
        """
        从磁盘加载会话数据

        Returns:
            会话数据字典

        Raises:
            SessionLoadError: 加载失败时抛出
        """
        if not self.sessions_file.exists():
            logger.info("会话文件不存在，将创建新文件")
            return {}

        try:
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载 {len(data)} 个会话")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"会话文件格式错误: {e}")
            raise SessionLoadError(f"会话文件格式错误: {e}")
        except IOError as e:
            logger.error(f"读取会话文件失败: {e}")
            raise SessionLoadError(f"读取会话文件失败: {e}")

    def save_sessions(self, data: Dict) -> None:
        """
        保存会话数据到磁盘

        Args:
            data: 会话数据字典

        Raises:
            SessionSaveError: 保存失败时抛出
        """
        try:
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"成功保存 {len(data)} 个会话")
        except IOError as e:
            logger.error(f"保存会话文件失败: {e}")
            raise SessionSaveError(f"保存会话文件失败: {e}")
        except TypeError as e:
            logger.error(f"会话数据序列化失败: {e}")
            raise SessionSaveError(f"会话数据序列化失败: {e}")


class ConversationStore:
    """
    对话存储管理器 - 单例模式

    提供对话会话的创建、查询、更新、删除功能。
    支持持久化存储和自动恢复。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """初始化存储管理器"""
        storage_dir = getattr(
            settings, 'CONVERSATION_STORAGE_DIR',
            os.path.join(settings.VECTOR_DB_PATH, "conversations")
        )

        self.file_manager = SessionFileManager(storage_dir)
        self.classifier = QuestionClassifier()
        self.sessions: Dict[str, ConversationSession] = {}

        self._load_sessions()

    def _load_sessions(self) -> None:
        """加载会话数据"""
        try:
            data = self.file_manager.load_sessions()
            self._parse_sessions_data(data)
        except SessionLoadError as e:
            logger.warning(f"会话加载失败，使用空数据: {e}")
            self.sessions = {}

    def _parse_sessions_data(self, data: Dict) -> None:
        """
        解析会话数据

        Args:
            data: 原始会话数据字典
        """
        for session_id, session_data in data.items():
            try:
                turns = self._parse_turns(session_data.get('turns', []))
                session = ConversationSession(
                    id=session_data['id'],
                    title=session_data['title'],
                    created_at=session_data['created_at'],
                    updated_at=session_data['updated_at'],
                    turns=turns,
                    metadata=session_data.get('metadata', {})
                )

                if session.validate():
                    self.sessions[session_id] = session
                else:
                    logger.warning(f"会话数据验证失败，跳过: {session_id}")

            except KeyError as e:
                logger.error(f"会话数据缺少必要字段 {e}，跳过: {session_id}")
            except Exception as e:
                logger.error(f"解析会话数据异常: {e}，跳过: {session_id}")

        logger.info(f"成功解析 {len(self.sessions)} 个会话")

    def _parse_turns(self, turns_data: List[Dict]) -> List[ConversationTurn]:
        """解析对话轮次数据"""
        turns = []
        for turn_data in turns_data:
            try:
                turn = ConversationTurn(**turn_data)
                if turn.validate():
                    turns.append(turn)
            except Exception as e:
                logger.warning(f"解析对话轮次失败: {e}")
        return turns

    def _save_sessions(self) -> None:
        """保存会话数据到磁盘"""
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

        try:
            self.file_manager.save_sessions(data)
        except SessionSaveError as e:
            logger.error(f"保存会话失败: {e}")

    def create_session(self, title: str = None) -> str:
        """
        创建新对话会话

        Args:
            title: 会话标题，可选

        Returns:
            新创建的会话ID

        Raises:
            SessionSaveError: 保存失败时抛出
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

        logger.info(f"创建新会话: {session_id}")
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
            question: 用户问题
            answer: 系统回答
            sources: 参考来源列表
            response_time_ms: 响应时间(毫秒)
            is_code_query: 是否为代码查询

        Returns:
            创建的对话轮次对象

        Raises:
            SessionNotFoundError: 会话不存在时抛出
        """
        if session_id not in self.sessions:
            logger.warning(f"会话不存在，自动创建: {session_id}")
            session_id = self.create_session()

        turn = ConversationTurn(
            id=str(uuid.uuid4()),
            session_id=session_id,
            question=question,
            answer=answer,
            sources=sources if sources else [],
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time_ms,
            is_code_query=is_code_query,
            category=self.classifier.classify(question),
            quality_score=0.0
        )

        self.sessions[session_id].turns.append(turn)
        self.sessions[session_id].updated_at = datetime.now().isoformat()
        self._save_sessions()

        logger.debug(f"添加对话轮次: {turn.id}")
        return turn

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        获取会话

        Args:
            session_id: 会话ID

        Returns:
            会话对象，不存在则返回None
        """
        return self.sessions.get(session_id)

    def get_all_sessions(self) -> List[Dict]:
        """
        获取所有会话列表

        Returns:
            会话摘要列表
        """
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
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            删除成功返回True，会话不存在返回False
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            logger.info(f"删除会话: {session_id}")
            return True

        logger.warning(f"尝试删除不存在的会话: {session_id}")
        return False

    def update_turn_quality_score(self, turn_id: str, score: float) -> bool:
        """
        更新对话轮次的质量评分

        Args:
            turn_id: 对话轮次ID
            score: 质量评分(0-100)

        Returns:
            更新成功返回True，未找到返回False
        """
        score = max(0.0, min(100.0, score))

        for session in self.sessions.values():
            for turn in session.turns:
                if turn.id == turn_id:
                    turn.quality_score = score
                    self._save_sessions()
                    logger.debug(f"更新质量评分: {turn_id} -> {score}")
                    return True

        logger.warning(f"未找到对话轮次: {turn_id}")
        return False

    def get_session_count(self) -> int:
        """获取会话总数"""
        return len(self.sessions)

    def get_total_turns(self) -> int:
        """获取所有对话轮次总数"""
        return sum(len(s.turns) for s in self.sessions.values())


# 全局单例实例
_conversation_store = None


def get_conversation_store() -> ConversationStore:
    """
    获取 ConversationStore 单例实例

    Returns:
        ConversationStore 实例
    """
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store
