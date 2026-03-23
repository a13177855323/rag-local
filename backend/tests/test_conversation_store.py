# encoding: utf-8
"""
对话存储模块单元测试
测试ConversationStore类及其相关数据结构
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# 添加项目路径到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.services.conversation_store import (
    ConversationStore, ConversationTurn, ConversationSession,
    SourceReference, QuestionCategory, QualityLevel,
    get_conversation_store
)


class TestSourceReference(unittest.TestCase):
    """来源引用数据结构测试"""

    def test_source_reference_creation(self):
        """测试SourceReference对象创建"""
        ref = SourceReference(
            filename="test.pdf",
            content="这是测试内容",
            similarity=0.85,
            chunk_id=1
        )

        self.assertEqual(ref.filename, "test.pdf")
        self.assertEqual(ref.content, "这是测试内容")
        self.assertEqual(ref.similarity, 0.85)
        self.assertEqual(ref.chunk_id, 1)

    def test_source_reference_from_dict(self):
        """测试从字典创建SourceReference"""
        data = {
            "filename": "test.md",
            "content": "字典内容",
            "similarity": 0.75,
            "chunk_id": 5
        }

        ref = SourceReference.from_dict(data)
        self.assertEqual(ref.filename, "test.md")
        self.assertEqual(ref.content, "字典内容")
        self.assertEqual(ref.similarity, 0.75)
        self.assertEqual(ref.chunk_id, 5)

    def test_source_reference_to_dict(self):
        """测试转换为字典"""
        ref = SourceReference(
            filename="test.txt",
            content="转换测试",
            similarity=0.9,
            chunk_id=3
        )

        ref_dict = ref.to_dict()
        self.assertIsInstance(ref_dict, dict)
        self.assertEqual(ref_dict["filename"], "test.txt")
        self.assertEqual(ref_dict["content"], "转换测试")


class TestQuestionCategory(unittest.TestCase):
    """问题分类枚举测试"""

    def test_category_values(self):
        """测试分类值"""
        self.assertEqual(QuestionCategory.CODE.value, "代码问题")
        self.assertEqual(QuestionCategory.CONCEPT.value, "概念解释")
        self.assertEqual(QuestionCategory.HOW_TO.value, "操作指南")
        self.assertEqual(QuestionCategory.DEBUG.value, "调试排错")
        self.assertEqual(QuestionCategory.COMPARISON.value, "对比分析")
        self.assertEqual(QuestionCategory.OTHER.value, "其他")

    def test_from_string(self):
        """测试从字符串转换"""
        self.assertEqual(
            QuestionCategory.from_string("代码问题"),
            QuestionCategory.CODE
        )
        self.assertEqual(
            QuestionCategory.from_string("未知分类"),
            QuestionCategory.OTHER
        )


class TestQualityLevel(unittest.TestCase):
    """质量等级枚举测试"""

    def test_quality_levels(self):
        """测试质量等级值"""
        self.assertEqual(QualityLevel.EXCELLENT.value, "优秀")
        self.assertEqual(QualityLevel.GOOD.value, "良好")
        self.assertEqual(QualityLevel.FAIR.value, "一般")
        self.assertEqual(QualityLevel.POOR.value, "较差")
        self.assertEqual(QualityLevel.UNKNOWN.value, "未知")


class TestConversationTurn(unittest.TestCase):
    """对话轮次数据结构测试"""

    def test_turn_creation(self):
        """测试ConversationTurn对象创建"""
        sources = [
            SourceReference(filename="doc1.pdf", content="内容1"),
            SourceReference(filename="doc2.pdf", content="内容2")
        ]

        turn = ConversationTurn(
            id="turn_123",
            session_id="session_456",
            question="Python是什么？",
            answer="Python是一种编程语言...",
            sources=sources,
            response_time_ms=1500,
            is_code_query=False
        )

        self.assertEqual(turn.id, "turn_123")
        self.assertEqual(turn.session_id, "session_456")
        self.assertEqual(turn.question, "Python是什么？")
        self.assertEqual(turn.answer, "Python是一种编程语言...")
        self.assertEqual(len(turn.sources), 2)
        self.assertEqual(turn.response_time_ms, 1500)
        self.assertFalse(turn.is_code_query)

    def test_turn_from_dict(self):
        """测试从字典创建ConversationTurn"""
        data = {
            "id": "test_id",
            "session_id": "test_session",
            "question": "测试问题？",
            "answer": "测试回答",
            "sources": [
                {"filename": "test.pdf", "content": "测试内容", "similarity": 0.8}
            ],
            "response_time_ms": 2000,
            "is_code_query": True,
            "category": "代码问题"
        }

        turn = ConversationTurn.from_dict(data)
        self.assertEqual(turn.id, "test_id")
        self.assertEqual(turn.session_id, "test_session")
        self.assertEqual(turn.question, "测试问题？")
        self.assertEqual(turn.category, QuestionCategory.CODE)
        self.assertTrue(turn.is_code_query)

    def test_get_word_count(self):
        """测试字数统计"""
        turn = ConversationTurn(
            id="test",
            session_id="test",
            question="这是一个问题？",
            answer="这是一个回答。"
        )

        q_words, a_words = turn.get_word_count()
        self.assertGreater(q_words, 0)
        self.assertGreater(a_words, 0)

    def test_calculate_quality_score(self):
        """测试质量分数计算"""
        turn = ConversationTurn(
            id="test",
            session_id="test",
            question="如何编写Python代码？",
            answer="""这是一个详细的回答，包含了足够的内容。
            ```python
            print("Hello World")
            ```
            这样你就可以运行Python代码了。""",
            response_time_ms=1500,
            is_code_query=True
        )

        score = turn.calculate_quality_score()
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIsNotNone(turn.quality_level)


class TestConversationSession(unittest.TestCase):
    """对话会话数据结构测试"""

    def test_session_creation(self):
        """测试ConversationSession对象创建"""
        session = ConversationSession(
            id="session_123",
            title="测试对话",
            turns=[]
        )

        self.assertEqual(session.id, "session_123")
        self.assertEqual(session.title, "测试对话")
        self.assertEqual(len(session.turns), 0)

    def test_add_turn(self):
        """测试添加对话轮次"""
        session = ConversationSession(
            id="session_123",
            title="Untitled",
            turns=[]
        )

        turn = ConversationTurn(
            id="turn_1",
            session_id="session_123",
            question="测试问题",
            answer="测试回答"
        )

        session.add_turn(turn)
        self.assertEqual(len(session.turns), 1)
        # 第一轮对话应该自动设置标题
        self.assertNotEqual(session.title, "Untitled")

    def test_get_turn_by_id(self):
        """测试根据ID获取轮次"""
        session = ConversationSession(
            id="session_123",
            title="测试",
            turns=[]
        )

        turn1 = ConversationTurn(
            id="turn_1",
            session_id="session_123",
            question="问题1",
            answer="回答1"
        )

        turn2 = ConversationTurn(
            id="turn_2",
            session_id="session_123",
            question="问题2",
            answer="回答2"
        )

        session.add_turn(turn1)
        session.add_turn(turn2)

        found = session.get_turn_by_id("turn_1")
        self.assertIsNotNone(found)
        self.assertEqual(found.question, "问题1")

        not_found = session.get_turn_by_id("nonexistent")
        self.assertIsNone(not_found)

    def test_get_statistics(self):
        """测试会话统计信息"""
        session = ConversationSession(
            id="session_123",
            title="测试统计",
            turns=[]
        )

        # 添加多个轮次
        for i in range(3):
            turn = ConversationTurn(
                id=f"turn_{i}",
                session_id="session_123",
                question=f"问题{i}",
                answer=f"回答{i}",
                response_time_ms=1000 + i * 500,
                is_code_query=(i % 2 == 0)  # 偶数是代码查询
            )
            turn.calculate_quality_score()
            session.add_turn(turn)

        stats = session.get_statistics()
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats.get("total_turns"), 3)
        self.assertIn("code_query_ratio", stats)
        self.assertIn("avg_response_time_ms", stats)
        self.assertIn("avg_quality_score", stats)


class TestConversationStore(unittest.TestCase):
    """对话存储管理器测试"""

    def setUp(self):
        """测试前准备"""
        # 创建临时目录用于测试
        self.test_dir = tempfile.mkdtemp()

        # 使用patch修改存储路径
        with patch.object(ConversationStore, '_initialized', False):
            with patch.object(ConversationStore, '_instance', None):
                self.store = ConversationStore()
                # 修改存储路径到临时目录
                self.store.storage_dir = self.test_dir
                self.store.sessions_file = os.path.join(self.test_dir, "sessions.json")
                self.store.backup_dir = os.path.join(self.test_dir, "backups")
                os.makedirs(self.store.backup_dir, exist_ok=True)
                self.store.sessions = {}

    def tearDown(self):
        """测试后清理"""
        # 移除临时目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_singleton_pattern(self):
        """测试单例模式"""
        store1 = get_conversation_store()
        store2 = get_conversation_store()
        self.assertIs(store1, store2)

    def test_create_session(self):
        """测试创建会话"""
        session_id = self.store.create_session(title="测试会话")
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.store.sessions)

        session = self.store.sessions[session_id]
        self.assertEqual(session.title, "测试会话")

    def test_add_turn(self):
        """测试添加对话轮次"""
        session_id = self.store.create_session()

        sources = [{"filename": "test.pdf", "content": "参考内容"}]
        turn = self.store.add_turn(
            session_id=session_id,
            question="Python如何安装？",
            answer="可以通过pip安装...",
            sources=sources,
            response_time_ms=1500,
            is_code_query=True
        )

        self.assertIsNotNone(turn)
        self.assertEqual(len(self.store.sessions[session_id].turns), 1)

    def test_get_session(self):
        """测试获取会话"""
        session_id = self.store.create_session(title="获取测试")
        session = self.store.get_session(session_id)

        self.assertIsNotNone(session)
        self.assertEqual(session.title, "获取测试")

        # 测试不存在的会话
        non_existent = self.store.get_session("nonexistent_id")
        self.assertIsNone(non_existent)

    def test_get_all_sessions(self):
        """测试获取所有会话"""
        # 创建几个会话
        for i in range(3):
            self.store.create_session(title=f"会话{i}")

        sessions = self.store.get_all_sessions()
        self.assertEqual(len(sessions), 3)

    def test_search_sessions(self):
        """测试搜索会话"""
        # 创建几个会话
        self.store.create_session(title="Python编程入门")
        self.store.create_session(title="Java编程基础")
        self.store.create_session(title="数据科学")

        # 添加带内容的轮次用于搜索
        for sid in self.store.sessions.keys():
            self.store.add_turn(
                session_id=sid,
                question="测试问题",
                answer="测试回答",
                sources=[]
            )

        # 搜索"Python"
        results = self.store.search_sessions("Python")
        self.assertGreaterEqual(len(results), 1)

    def test_update_session_title(self):
        """测试更新会话标题"""
        session_id = self.store.create_session(title="旧标题")
        result = self.store.update_session_title(session_id, "新标题")

        self.assertTrue(result)
        self.assertEqual(self.store.sessions[session_id].title, "新标题")

        # 测试不存在的会话
        result = self.store.update_session_title("nonexistent", "标题")
        self.assertFalse(result)

    def test_delete_session(self):
        """测试删除会话"""
        session_id = self.store.create_session(title="待删除")
        self.assertIn(session_id, self.store.sessions)

        result = self.store.delete_session(session_id)
        self.assertTrue(result)
        self.assertNotIn(session_id, self.store.sessions)

        # 测试删除不存在的会话
        result = self.store.delete_session("nonexistent")
        self.assertFalse(result)

    def test_global_statistics(self):
        """测试全局统计"""
        # 创建一些带数据的会话
        for i in range(2):
            session_id = self.store.create_session(title=f"会话{i}")
            for j in range(3):
                self.store.add_turn(
                    session_id=session_id,
                    question=f"问题{i}-{j}",
                    answer=f"回答{i}-{j}",
                    sources=[],
                    is_code_query=(i == 0)  # 第一个会话是代码查询
                )

        stats = self.store.get_global_statistics()
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats.get("total_sessions"), 2)
        self.assertEqual(stats.get("total_turns"), 6)
        self.assertIn("code_query_ratio", stats)

    def test_export_session(self):
        """测试导出会话"""
        session_id = self.store.create_session(title="导出测试")
        self.store.add_turn(
            session_id=session_id,
            question="导出测试问题",
            answer="导出测试回答",
            sources=[{"filename": "doc.pdf", "content": "内容"}]
        )

        # 导出为JSON格式字符串
        json_export = self.store.export_session(session_id, format_type='json')
        self.assertIsInstance(json_export, str)

        # 导出为字典
        dict_export = self.store.export_session(session_id, format_type='dict')
        self.assertIsInstance(dict_export, dict)

    def test_classify_question(self):
        """测试问题分类"""
        # 代码问题
        code_question = "这段Python代码有什么问题？"
        category = self.store._classify_question(code_question)
        self.assertEqual(category, QuestionCategory.CODE)

        # 概念问题
        concept_question = "什么是人工智能？"
        category = self.store._classify_question(concept_question)
        self.assertEqual(category, QuestionCategory.CONCEPT)

        # HowTo问题
        howto_question = "如何安装Python？"
        category = self.store._classify_question(howto_question)
        self.assertEqual(category, QuestionCategory.HOW_TO)

        # 调试问题
        debug_question = "这个报错是什么意思？"
        category = self.store._classify_question(debug_question)
        self.assertEqual(category, QuestionCategory.DEBUG)

        # 对比问题
        comparison_question = "Python和Java的区别？"
        category = self.store._classify_question(comparison_question)
        self.assertEqual(category, QuestionCategory.COMPARISON)


if __name__ == '__main__':
    unittest.main(verbosity=2)
