"""
对话历史存储模块
支持对话记录的存储、查询、导出和分析

重构改进:
- 模块化设计: 分类逻辑独立为子函数
- 配置管理: 使用 ConversationConfig 集中管理配置
- 异常处理: 完善的异常捕获和日志记录
- PEP8 规范: 统一的命名和代码风格
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from backend.utils.logger import get_logger, log_info, log_error, log_warning
from backend.utils.conversation_config import get_config


# 模块常量
MODULE_NAME = "ConversationStore"


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


class ClassificationEngine:
    """问题分类引擎"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()

    def classify(self, question: str) -> str:
        """
        对问题进行分类

        Args:
            question: 问题文本

        Returns:
            分类标签
        """
        if not question or not isinstance(question, str):
            log_warning(
                MODULE_NAME,
                "Invalid question for classification",
                {"question_type": type(question).__name__}
            )
            return QuestionCategory.OTHER.value

        question_lower = question.lower()

        # 按优先级依次检查
        if self._is_code_question(question_lower):
            return QuestionCategory.CODE.value

        if self._is_debug_question(question_lower):
            return QuestionCategory.DEBUG.value

        if self._is_comparison_question(question_lower):
            return QuestionCategory.COMPARISON.value

        if self._is_how_to_question(question_lower):
            return QuestionCategory.HOW_TO.value

        if self._is_concept_question(question_lower):
            return QuestionCategory.CONCEPT.value

        return QuestionCategory.OTHER.value

    def _is_code_question(self, question_lower: str) -> bool:
        """检查是否为代码相关问题"""
        keywords = self.config.get_category_keywords('code')
        return any(kw in question_lower for kw in keywords)

    def _is_how_to_question(self, question_lower: str) -> bool:
        """检查是否为操作指南问题"""
        keywords = self.config.get_category_keywords('how_to')
        return any(kw in question_lower for kw in keywords)

    def _is_debug_question(self, question_lower: str) -> bool:
        """检查是否为调试排错问题"""
        keywords = self.config.get_category_keywords('debug')
        return any(kw in question_lower for kw in keywords)

    def _is_comparison_question(self, question_lower: str) -> bool:
        """检查是否为对比分析问题"""
        keywords = self.config.get_category_keywords('compare')
        return any(kw in question_lower for kw in keywords)

    def _is_concept_question(self, question_lower: str) -> bool:
        """检查是否为概念解释问题"""
        keywords = self.config.get_category_keywords('concept')
        return any(kw in question_lower for kw in keywords)


class StorageManager:
    """存储管理器 - 处理文件读写操作"""

    def __init__(self, storage_dir: str, sessions_filename: str):
        self.storage_dir = storage_dir
        self.sessions_file = os.path.join(storage_dir, sessions_filename)
        self.logger = get_logger()
        self._ensure_storage_exists()

    def _ensure_storage_exists(self):
        """确保存储目录存在"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            log_info(
                MODULE_NAME,
                "Storage directory ensured",
                {"path": self.storage_dir}
            )
        except OSError as e:
            log_error(
                MODULE_NAME,
                "Failed to create storage directory",
                {"path": self.storage_dir, "error": str(e)}
            )
            raise StorageError(f"无法创建存储目录: {e}") from e

    def load_sessions(self) -> Dict[str, Dict]:
        """
        从磁盘加载会话数据

        Returns:
            会话数据字典

        Raises:
            StorageError: 存储操作失败
        """
        if not os.path.exists(self.sessions_file):
            log_info(MODULE_NAME, "Sessions file not found, starting fresh")
            return {}

        try:
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                log_info(
                    MODULE_NAME,
                    "Sessions loaded successfully",
                    {"count": len(data)}
                )
                return data

        except json.JSONDecodeError as e:
            log_error(
                MODULE_NAME,
                "JSON decode error when loading sessions",
                {"file": self.sessions_file, "error": str(e)}
            )
            raise StorageError(f"会话数据格式错误: {e}") from e

        except FileNotFoundError as e:
            log_warning(
                MODULE_NAME,
                "Sessions file not found",
                {"file": self.sessions_file}
            )
            return {}

        except PermissionError as e:
            log_error(
                MODULE_NAME,
                "Permission denied when loading sessions",
                {"file": self.sessions_file}
            )
            raise StorageError(f"无法读取会话文件: {e}") from e

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Unexpected error when loading sessions",
                {"error": str(e)}
            )
            raise StorageError(f"加载会话数据失败: {e}") from e

    def save_sessions(self, sessions_data: Dict[str, Dict]):
        """
        保存会话数据到磁盘

        Args:
            sessions_data: 会话数据字典

        Raises:
            StorageError: 存储操作失败
        """
        try:
            # 写入临时文件，然后原子替换
            temp_file = f"{self.sessions_file}.tmp"

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2)

            # 原子替换
            os.replace(temp_file, self.sessions_file)

            log_info(
                MODULE_NAME,
                "Sessions saved successfully",
                {"count": len(sessions_data)}
            )

        except PermissionError as e:
            log_error(
                MODULE_NAME,
                "Permission denied when saving sessions",
                {"file": self.sessions_file}
            )
            raise StorageError(f"无法写入会话文件: {e}") from e

        except OSError as e:
            log_error(
                MODULE_NAME,
                "OS error when saving sessions",
                {"file": self.sessions_file, "error": str(e)}
            )
            raise StorageError(f"保存会话数据失败: {e}") from e

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Unexpected error when saving sessions",
                {"error": str(e)}
            )
            raise StorageError(f"保存会话数据失败: {e}") from e


class StorageError(Exception):
    """存储操作异常"""
    pass


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
        self.config = get_config()
        self.logger = get_logger()
        self.classifier = ClassificationEngine()

        # 初始化存储管理器
        self.storage = StorageManager(
            storage_dir=self.config.storage_dir,
            sessions_filename=self.config.sessions_filename
        )

        # 加载会话数据
        self.sessions: Dict[str, ConversationSession] = {}
        self._load_sessions()

    def _load_sessions(self):
        """加载会话数据"""
        try:
            data = self.storage.load_sessions()

            for session_id, session_data in data.items():
                try:
                    turns = [
                        ConversationTurn(**turn)
                        for turn in session_data.get('turns', [])
                    ]

                    self.sessions[session_id] = ConversationSession(
                        id=session_data['id'],
                        title=session_data['title'],
                        created_at=session_data['created_at'],
                        updated_at=session_data['updated_at'],
                        turns=turns,
                        metadata=session_data.get('metadata', {})
                    )

                except (KeyError, TypeError) as e:
                    log_warning(
                        MODULE_NAME,
                        "Failed to parse session data",
                        {"session_id": session_id, "error": str(e)}
                    )
                    continue

            log_info(
                MODULE_NAME,
                "Sessions initialized",
                {"count": len(self.sessions)}
            )

        except StorageError as e:
            log_error(MODULE_NAME, "Failed to load sessions", {"error": str(e)})
            self.sessions = {}

    def _save_sessions(self):
        """保存会话数据"""
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

            self.storage.save_sessions(data)

        except StorageError as e:
            log_error(MODULE_NAME, "Failed to save sessions", {"error": str(e)})
            raise

    def create_session(self, title: Optional[str] = None) -> str:
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

        try:
            self._save_sessions()
            log_info(
                MODULE_NAME,
                "Session created",
                {"session_id": session_id, "title": title}
            )
        except StorageError as e:
            log_error(
                MODULE_NAME,
                "Failed to save new session",
                {"session_id": session_id, "error": str(e)}
            )

        return session_id

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict],
        response_time_ms: int = 0,
        is_code_query: bool = False
    ) -> Optional[ConversationTurn]:
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
            创建的对话轮次，失败返回 None
        """
        # 验证会话存在
        if session_id not in self.sessions:
            log_warning(
                MODULE_NAME,
                "Session not found, creating new",
                {"session_id": session_id}
            )
            session_id = self.create_session()

        # 验证输入
        if not question or not isinstance(question, str):
            log_error(
                MODULE_NAME,
                "Invalid question",
                {"session_id": session_id, "type": type(question).__name__}
            )
            return None

        try:
            turn = ConversationTurn(
                id=str(uuid.uuid4()),
                session_id=session_id,
                question=question,
                answer=answer,
                sources=sources or [],
                timestamp=datetime.now().isoformat(),
                response_time_ms=max(0, response_time_ms),
                is_code_query=is_code_query,
                category=self.classifier.classify(question),
                quality_score=0.0
            )

            self.sessions[session_id].turns.append(turn)
            self.sessions[session_id].updated_at = datetime.now().isoformat()

            self._save_sessions()

            log_info(
                MODULE_NAME,
                "Turn added",
                {
                    "session_id": session_id,
                    "turn_id": turn.id,
                    "category": turn.category
                }
            )

            return turn

        except Exception as e:
            log_error(
                MODULE_NAME,
                "Failed to add turn",
                {"session_id": session_id, "error": str(e)}
            )
            return None

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        获取会话

        Args:
            session_id: 会话ID

        Returns:
            会话对象，不存在返回 None
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
            是否删除成功
        """
        if session_id not in self.sessions:
            log_warning(
                MODULE_NAME,
                "Cannot delete non-existent session",
                {"session_id": session_id}
            )
            return False

        try:
            del self.sessions[session_id]
            self._save_sessions()

            log_info(
                MODULE_NAME,
                "Session deleted",
                {"session_id": session_id}
            )
            return True

        except StorageError as e:
            log_error(
                MODULE_NAME,
                "Failed to delete session",
                {"session_id": session_id, "error": str(e)}
            )
            return False

    def update_turn_quality_score(self, turn_id: str, score: float) -> bool:
        """
        更新对话轮次的质量评分

        Args:
            turn_id: 轮次ID
            score: 质量评分

        Returns:
            是否更新成功
        """
        for session in self.sessions.values():
            for turn in session.turns:
                if turn.id == turn_id:
                    turn.quality_score = max(0.0, min(100.0, score))
                    try:
                        self._save_sessions()
                        log_info(
                            MODULE_NAME,
                            "Quality score updated",
                            {"turn_id": turn_id, "score": score}
                        )
                        return True
                    except StorageError as e:
                        log_error(
                            MODULE_NAME,
                            "Failed to save quality score",
                            {"turn_id": turn_id, "error": str(e)}
                        )
                        return False

        log_warning(
            MODULE_NAME,
            "Turn not found for quality update",
            {"turn_id": turn_id}
        )
        return False


# 全局单例实例
_conversation_store = None


def get_conversation_store() -> ConversationStore:
    """获取 ConversationStore 单例实例"""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store
