#!/usr/bin/env python
"""
多轮对话质量评估工具

评估RAG系统在多轮对话场景下的表现，包括：
- 指代消解能力
- 上下文关联理解
- 主题切换与回归
- 信息补全质量

Author: RAG System Test Engineer
Date: 2026-03-24
"""

import os
import sys
import json
import time
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backend.services.rag_service import RAGService
    from backend.services.conversation_store import ConversationStore, get_conversation_store
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    print("警告: 无法导入后端模块，将使用模拟模式运行")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class TestScenarioType(Enum):
    COREFERENCE_RESOLUTION = "指代消解"
    CONTEXT_ASSOCIATION = "上下文关联"
    TOPIC_SWITCH_RETURN = "主题切换与回归"
    INFORMATION_COMPLETION = "信息补全"


@dataclass
class ConversationTurn:
    turn_id: int
    question: str
    expected_keywords: List[str]
    expected_context_refs: List[str]
    scenario_type: TestScenarioType
    description: str = ""
    validation_rules: Dict = field(default_factory=dict)


@dataclass
class TestScenario:
    scenario_id: str
    scenario_name: str
    scenario_type: TestScenarioType
    description: str
    turns: List[ConversationTurn]
    expected_flow: List[str] = field(default_factory=list)


@dataclass
class TurnEvaluationResult:
    turn_id: int
    question: str
    answer: str
    scenario_type: str
    keyword_match_score: float
    context_ref_score: float
    coherence_score: float
    issues: List[str] = field(default_factory=list)
    response_time_ms: int = 0


@dataclass
class ScenarioEvaluationResult:
    scenario_id: str
    scenario_name: str
    scenario_type: str
    total_turns: int
    avg_coherence_score: float
    avg_keyword_match: float
    avg_context_ref: float
    coreference_accuracy: float
    context_retention_score: float
    turn_results: List[TurnEvaluationResult] = field(default_factory=list)


@dataclass
class MultiTurnEvaluationReport:
    test_time: str
    total_scenarios: int
    total_turns: int
    overall_coherence_score: float
    overall_coreference_accuracy: float
    overall_context_retention: float
    overall_keyword_match: float
    by_scenario_type: Dict[str, Dict] = field(default_factory=dict)
    scenario_results: List[ScenarioEvaluationResult] = field(default_factory=list)


class TestScenarioGenerator:
    """多轮对话测试场景生成器"""
    
    def __init__(self):
        self.scenario_counter = 0
    
    def generate_all_scenarios(self) -> List[TestScenario]:
        """生成所有测试场景"""
        scenarios = []
        scenarios.extend(self._generate_coreference_scenarios())
        scenarios.extend(self._generate_context_association_scenarios())
        scenarios.extend(self._generate_topic_switch_scenarios())
        scenarios.extend(self._generate_info_completion_scenarios())
        return scenarios
    
    def _generate_coreference_scenarios(self) -> List[TestScenario]:
        """生成指代消解测试场景"""
        scenarios = []
        
        scenario1 = TestScenario(
            scenario_id="coref_001",
            scenario_name="代码指代消解测试",
            scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
            description="测试系统对代码相关指代词的理解能力",
            turns=[
                ConversationTurn(
                    turn_id=1,
                    question="什么是Python中的装饰器？",
                    expected_keywords=["装饰器", "decorator", "@", "函数"],
                    expected_context_refs=[],
                    scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
                    description="首次提问，建立上下文"
                ),
                ConversationTurn(
                    turn_id=2,
                    question="它有什么作用？",
                    expected_keywords=["装饰器", "功能", "作用", "修改", "扩展"],
                    expected_context_refs=["装饰器"],
                    scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
                    description="测试'它'指代'装饰器'"
                ),
                ConversationTurn(
                    turn_id=3,
                    question="能给个例子吗？",
                    expected_keywords=["@", "def", "例子", "示例", "代码"],
                    expected_context_refs=["装饰器", "例子"],
                    scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
                    description="测试省略主语的指代"
                ),
                ConversationTurn(
                    turn_id=4,
                    question="这个例子中的@符号是什么意思？",
                    expected_keywords=["@", "语法", "装饰器", "应用"],
                    expected_context_refs=["例子", "@"],
                    scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
                    description="测试'这个例子'的指代"
                ),
                ConversationTurn(
                    turn_id=5,
                    question="前者可以被多个函数使用吗？",
                    expected_keywords=["装饰器", "多个", "函数", "复用"],
                    expected_context_refs=["装饰器"],
                    scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
                    description="测试'前者'指代'装饰器'"
                )
            ],
            expected_flow=["装饰器定义", "作用说明", "示例展示", "语法解释", "复用性"]
        )
        scenarios.append(scenario1)
        
        scenario2 = TestScenario(
            scenario_id="coref_002",
            scenario_name="概念指代消解测试",
            scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
            description="测试系统对概念性指代词的理解",
            turns=[
                ConversationTurn(
                    turn_id=1,
                    question="什么是向量数据库？",
                    expected_keywords=["向量", "数据库", "存储", "检索"],
                    expected_context_refs=[],
                    scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
                    description="建立向量数据库概念"
                ),
                ConversationTurn(
                    turn_id=2,
                    question="它与传统的SQL数据库有什么区别？",
                    expected_keywords=["区别", "SQL", "向量", "检索", "结构化"],
                    expected_context_refs=["向量数据库", "SQL数据库"],
                    scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
                    description="测试'它'指代'向量数据库'"
                ),
                ConversationTurn(
                    turn_id=3,
                    question="上述区别在实际应用中有什么影响？",
                    expected_keywords=["应用", "影响", "场景", "性能"],
                    expected_context_refs=["区别"],
                    scenario_type=TestScenarioType.COREFERENCE_RESOLUTION,
                    description="测试'上述区别'的指代"
                )
            ],
            expected_flow=["概念定义", "对比分析", "应用影响"]
        )
        scenarios.append(scenario2)
        
        return scenarios
    
    def _generate_context_association_scenarios(self) -> List[TestScenario]:
        """生成上下文关联测试场景"""
        scenarios = []
        
        scenario1 = TestScenario(
            scenario_id="ctx_001",
            scenario_name="代码上下文关联测试",
            scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
            description="测试系统对代码相关上下文的理解和关联能力",
            turns=[
                ConversationTurn(
                    turn_id=1,
                    question="如何在Python中读取文件？",
                    expected_keywords=["open", "read", "文件", "Python"],
                    expected_context_refs=[],
                    scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
                    description="建立文件操作上下文"
                ),
                ConversationTurn(
                    turn_id=2,
                    question="读取大文件时有什么注意事项？",
                    expected_keywords=["大文件", "内存", "缓冲", "逐行", "chunk"],
                    expected_context_refs=["读取文件"],
                    scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
                    description="基于之前内容深入提问"
                ),
                ConversationTurn(
                    turn_id=3,
                    question="如果文件不存在会发生什么？",
                    expected_keywords=["异常", "FileNotFoundError", "错误", "处理"],
                    expected_context_refs=["文件", "读取"],
                    scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
                    description="测试异常情况理解"
                ),
                ConversationTurn(
                    turn_id=4,
                    question="如何优雅地处理这种情况？",
                    expected_keywords=["try", "except", "处理", "优雅"],
                    expected_context_refs=["异常", "文件不存在"],
                    scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
                    description="测试'这种情况'指代异常"
                ),
                ConversationTurn(
                    turn_id=5,
                    question="with语句能解决前面提到的问题吗？",
                    expected_keywords=["with", "上下文管理", "自动关闭", "资源"],
                    expected_context_refs=["问题", "文件操作"],
                    scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
                    description="测试'前面提到的问题'关联"
                )
            ],
            expected_flow=["基础操作", "进阶注意", "异常处理", "优雅方案", "with语句"]
        )
        scenarios.append(scenario1)
        
        scenario2 = TestScenario(
            scenario_id="ctx_002",
            scenario_name="RAG系统上下文关联测试",
            scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
            description="测试RAG相关概念的上下文关联",
            turns=[
                ConversationTurn(
                    turn_id=1,
                    question="RAG系统是如何工作的？",
                    expected_keywords=["检索", "生成", "RAG", "文档", "向量"],
                    expected_context_refs=[],
                    scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
                    description="建立RAG概念"
                ),
                ConversationTurn(
                    turn_id=2,
                    question="检索阶段使用什么技术？",
                    expected_keywords=["向量", "检索", "相似度", "embedding"],
                    expected_context_refs=["RAG", "检索"],
                    scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
                    description="深入检索阶段"
                ),
                ConversationTurn(
                    turn_id=3,
                    question="生成阶段呢？",
                    expected_keywords=["生成", "LLM", "上下文", "回答"],
                    expected_context_refs=["RAG", "生成"],
                    scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
                    description="测试'呢'省略关联"
                ),
                ConversationTurn(
                    turn_id=4,
                    question="这两个阶段如何配合？",
                    expected_keywords=["配合", "检索", "生成", "流程"],
                    expected_context_refs=["检索阶段", "生成阶段"],
                    scenario_type=TestScenarioType.CONTEXT_ASSOCIATION,
                    description="测试'这两个阶段'指代"
                )
            ],
            expected_flow=["整体流程", "检索详解", "生成详解", "阶段配合"]
        )
        scenarios.append(scenario2)
        
        return scenarios
    
    def _generate_topic_switch_scenarios(self) -> List[TestScenario]:
        """生成主题切换与回归测试场景"""
        scenarios = []
        
        scenario1 = TestScenario(
            scenario_id="topic_001",
            scenario_name="主题切换回归测试",
            scenario_type=TestScenarioType.TOPIC_SWITCH_RETURN,
            description="测试系统在主题切换后能否正确回归之前话题",
            turns=[
                ConversationTurn(
                    turn_id=1,
                    question="什么是机器学习？",
                    expected_keywords=["机器学习", "学习", "数据", "模型"],
                    expected_context_refs=[],
                    scenario_type=TestScenarioType.TOPIC_SWITCH_RETURN,
                    description="建立机器学习主题"
                ),
                ConversationTurn(
                    turn_id=2,
                    question="它有哪些主要类型？",
                    expected_keywords=["监督", "无监督", "强化", "类型"],
                    expected_context_refs=["机器学习"],
                    scenario_type=TestScenarioType.TOPIC_SWITCH_RETURN,
                    description="深入机器学习"
                ),
                ConversationTurn(
                    turn_id=3,
                    question="顺便问一下，Python中如何定义类？",
                    expected_keywords=["class", "Python", "定义", "类"],
                    expected_context_refs=[],
                    scenario_type=TestScenarioType.TOPIC_SWITCH_RETURN,
                    description="主题切换到Python"
                ),
                ConversationTurn(
                    turn_id=4,
                    question="类的继承怎么实现？",
                    expected_keywords=["继承", "class", "父类", "子类"],
                    expected_context_refs=["类", "Python"],
                    scenario_type=TestScenarioType.TOPIC_SWITCH_RETURN,
                    description="继续Python主题"
                ),
                ConversationTurn(
                    turn_id=5,
                    question="回到之前的话题，监督学习和无监督学习的区别是什么？",
                    expected_keywords=["监督", "无监督", "区别", "标签"],
                    expected_context_refs=["机器学习", "监督学习", "无监督学习"],
                    scenario_type=TestScenarioType.TOPIC_SWITCH_RETURN,
                    description="回归机器学习主题"
                ),
                ConversationTurn(
                    turn_id=6,
                    question="刚才说的第一种类型适合什么场景？",
                    expected_keywords=["监督学习", "场景", "分类", "回归"],
                    expected_context_refs=["监督学习", "第一种类型"],
                    scenario_type=TestScenarioType.TOPIC_SWITCH_RETURN,
                    description="测试'刚才说的'和'第一种类型'"
                )
            ],
            expected_flow=["机器学习", "类型", "Python类", "继承", "回归ML", "场景"]
        )
        scenarios.append(scenario1)
        
        return scenarios
    
    def _generate_info_completion_scenarios(self) -> List[TestScenario]:
        """生成信息补全测试场景"""
        scenarios = []
        
        scenario1 = TestScenario(
            scenario_id="info_001",
            scenario_name="代码信息补全测试",
            scenario_type=TestScenarioType.INFORMATION_COMPLETION,
            description="测试系统基于已有信息补全后续回答的能力",
            turns=[
                ConversationTurn(
                    turn_id=1,
                    question="我有一个包含用户信息的字典：{'name': '张三', 'age': 25}",
                    expected_keywords=["字典", "用户", "信息"],
                    expected_context_refs=[],
                    scenario_type=TestScenarioType.INFORMATION_COMPLETION,
                    description="提供初始信息"
                ),
                ConversationTurn(
                    turn_id=2,
                    question="如何获取用户的名字？",
                    expected_keywords=["name", "张三", "获取", "字典"],
                    expected_context_refs=["字典", "用户信息"],
                    scenario_type=TestScenarioType.INFORMATION_COMPLETION,
                    description="基于已有信息回答"
                ),
                ConversationTurn(
                    turn_id=3,
                    question="如果要添加一个新的字段'city'，应该怎么做？",
                    expected_keywords=["city", "添加", "字典", "键值"],
                    expected_context_refs=["字典", "用户信息"],
                    scenario_type=TestScenarioType.INFORMATION_COMPLETION,
                    description="扩展已有信息"
                ),
                ConversationTurn(
                    turn_id=4,
                    question="现在这个字典有哪些键？",
                    expected_keywords=["name", "age", "city", "键"],
                    expected_context_refs=["字典", "name", "age", "city"],
                    scenario_type=TestScenarioType.INFORMATION_COMPLETION,
                    description="测试信息融合"
                ),
                ConversationTurn(
                    turn_id=5,
                    question="如何遍历这个字典的所有键值对？",
                    expected_keywords=["遍历", "items", "for", "键值对"],
                    expected_context_refs=["字典"],
                    scenario_type=TestScenarioType.INFORMATION_COMPLETION,
                    description="基于完整信息操作"
                )
            ],
            expected_flow=["初始信息", "获取操作", "添加操作", "信息确认", "遍历操作"]
        )
        scenarios.append(scenario1)
        
        scenario2 = TestScenario(
            scenario_id="info_002",
            scenario_name="配置信息补全测试",
            scenario_type=TestScenarioType.INFORMATION_COMPLETION,
            description="测试系统对配置信息的记忆和补全",
            turns=[
                ConversationTurn(
                    turn_id=1,
                    question="我正在配置一个RAG系统，向量维度是1024，使用FAISS索引",
                    expected_keywords=["RAG", "向量", "1024", "FAISS"],
                    expected_context_refs=[],
                    scenario_type=TestScenarioType.INFORMATION_COMPLETION,
                    description="提供配置信息"
                ),
                ConversationTurn(
                    turn_id=2,
                    question="基于这个配置，我应该选择什么嵌入模型？",
                    expected_keywords=["嵌入", "模型", "1024", "维度"],
                    expected_context_refs=["1024", "向量维度"],
                    scenario_type=TestScenarioType.INFORMATION_COMPLETION,
                    description="基于配置推荐"
                ),
                ConversationTurn(
                    turn_id=3,
                    question="如果我想把top_k设置为5，需要修改哪里？",
                    expected_keywords=["top_k", "5", "配置", "检索"],
                    expected_context_refs=["RAG系统", "配置"],
                    scenario_type=TestScenarioType.INFORMATION_COMPLETION,
                    description="扩展配置"
                ),
                ConversationTurn(
                    turn_id=4,
                    question="总结一下我目前的配置",
                    expected_keywords=["1024", "FAISS", "top_k", "5", "向量"],
                    expected_context_refs=["向量维度", "FAISS", "top_k"],
                    scenario_type=TestScenarioType.INFORMATION_COMPLETION,
                    description="测试信息融合"
                )
            ],
            expected_flow=["初始配置", "模型推荐", "配置扩展", "配置总结"]
        )
        scenarios.append(scenario2)
        
        return scenarios


class MockRAGService:
    """模拟RAG服务（用于无后端时的测试）"""
    
    def __init__(self):
        self.sessions = {}
        self.turn_counter = {}
    
    def create_conversation_session(self, title: str = None) -> str:
        import uuid
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        self.turn_counter[session_id] = 0
        return session_id
    
    def query(self, question: str, session_id: str = None, **kwargs):
        if session_id not in self.sessions:
            session_id = self.create_conversation_session()
        
        self.turn_counter[session_id] += 1
        turn_num = self.turn_counter[session_id]
        
        mock_answers = {
            "装饰器": "Python装饰器是一种特殊的语法，使用@符号来修改或扩展函数的行为。装饰器本质上是一个函数，它接受一个函数作为参数并返回一个新的函数。",
            "向量数据库": "向量数据库是专门用于存储和检索向量嵌入的数据库系统。与传统SQL数据库不同，它使用向量相似度进行检索，适合语义搜索和推荐系统。",
            "机器学习": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测或决策，而无需显式编程。",
            "RAG": "RAG（检索增强生成）是一种结合检索和生成的技术。它首先从知识库检索相关文档，然后将这些文档作为上下文输入到LLM生成回答。",
            "Python": "Python是一种高级编程语言，以简洁易读著称。它支持多种编程范式，包括面向对象、函数式和过程式编程。",
        }
        
        answer = f"[模拟回答{turn_num}] 关于'{question[:20]}...'的回答。"
        for key, value in mock_answers.items():
            if key in question:
                answer = value
                break
        
        self.sessions[session_id].append({
            "question": question,
            "answer": answer
        })
        
        return {
            "answer": answer,
            "sources": [],
            "session_id": session_id
        }


class MultiTurnEvaluator:
    """多轮对话质量评估器"""
    
    PRONOUNS = ["它", "这个", "那个", "前者", "后者", "上述", "刚才", "之前", "前面", "以上"]
    CONTEXT_REF_PATTERNS = [
        r"它[是为]?",
        r"这个\w*",
        r"那个\w*",
        r"前者",
        r"后者",
        r"上述\w*",
        r"刚才\w*",
        r"之前\w*",
        r"前面\w*",
        r"回到\w*",
        r"第一种",
        r"第二种",
    ]
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock or not HAS_BACKEND
        
        if self.use_mock:
            self.rag_service = MockRAGService()
            logger.info("使用模拟模式运行评估")
        else:
            self.rag_service = RAGService()
            logger.info("使用实际后端运行评估")
        
        self.scenario_generator = TestScenarioGenerator()
        self.test_scenarios: List[TestScenario] = []
        self.evaluation_results: List[ScenarioEvaluationResult] = []
    
    def load_test_scenarios(self, scenarios: List[TestScenario] = None) -> List[TestScenario]:
        """加载测试场景"""
        if scenarios is None:
            self.test_scenarios = self.scenario_generator.generate_all_scenarios()
        else:
            self.test_scenarios = scenarios
        
        logger.info(f"加载了 {len(self.test_scenarios)} 个测试场景")
        return self.test_scenarios
    
    def load_scenarios_from_file(self, file_path: str) -> List[TestScenario]:
        """从文件加载测试场景"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenarios = []
        for s in data:
            turns = [
                ConversationTurn(
                    turn_id=t['turn_id'],
                    question=t['question'],
                    expected_keywords=t['expected_keywords'],
                    expected_context_refs=t['expected_context_refs'],
                    scenario_type=TestScenarioType(t['scenario_type']),
                    description=t.get('description', '')
                )
                for t in s['turns']
            ]
            scenarios.append(TestScenario(
                scenario_id=s['scenario_id'],
                scenario_name=s['scenario_name'],
                scenario_type=TestScenarioType(s['scenario_type']),
                description=s['description'],
                turns=turns,
                expected_flow=s.get('expected_flow', [])
            ))
        
        self.test_scenarios = scenarios
        return scenarios
    
    def save_scenarios_to_file(self, file_path: str):
        """保存测试场景到文件"""
        data = []
        for s in self.test_scenarios:
            data.append({
                'scenario_id': s.scenario_id,
                'scenario_name': s.scenario_name,
                'scenario_type': s.scenario_type.value,
                'description': s.description,
                'turns': [
                    {
                        'turn_id': t.turn_id,
                        'question': t.question,
                        'expected_keywords': t.expected_keywords,
                        'expected_context_refs': t.expected_context_refs,
                        'scenario_type': t.scenario_type.value,
                        'description': t.description
                    }
                    for t in s.turns
                ],
                'expected_flow': s.expected_flow
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试场景已保存到: {file_path}")
    
    def evaluate_turn(
        self,
        turn: ConversationTurn,
        answer: str,
        conversation_history: List[Dict]
    ) -> TurnEvaluationResult:
        """评估单轮对话"""
        issues = []
        
        keyword_match_score = self._calculate_keyword_match(
            answer, turn.expected_keywords
        )
        
        context_ref_score = self._calculate_context_reference(
            turn.question, answer, conversation_history, turn.expected_context_refs
        )
        
        coherence_score = self._calculate_coherence(
            turn.question, answer, conversation_history
        )
        
        if keyword_match_score < 0.5:
            issues.append(f"关键词匹配不足，期望关键词: {turn.expected_keywords}")
        
        if turn.scenario_type == TestScenarioType.COREFERENCE_RESOLUTION:
            if not self._check_pronoun_resolution(turn.question, answer, conversation_history):
                issues.append("指代消解失败，未能正确理解指代词")
        
        if turn.scenario_type == TestScenarioType.CONTEXT_ASSOCIATION:
            if context_ref_score < 0.3:
                issues.append("上下文关联不足，未能正确引用前文内容")
        
        if turn.scenario_type == TestScenarioType.TOPIC_SWITCH_RETURN:
            if "回到" in turn.question or "刚才" in turn.question:
                if context_ref_score < 0.4:
                    issues.append("主题回归失败，未能正确回到之前的话题")
        
        return TurnEvaluationResult(
            turn_id=turn.turn_id,
            question=turn.question,
            answer=answer,
            scenario_type=turn.scenario_type.value,
            keyword_match_score=keyword_match_score,
            context_ref_score=context_ref_score,
            coherence_score=coherence_score,
            issues=issues
        )
    
    def _calculate_keyword_match(self, answer: str, expected_keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        if not expected_keywords:
            return 1.0
        
        answer_lower = answer.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        return matched / len(expected_keywords)
    
    def _calculate_context_reference(
        self,
        question: str,
        answer: str,
        history: List[Dict],
        expected_refs: List[str]
    ) -> float:
        """计算上下文引用分数"""
        if not expected_refs or not history:
            return 1.0
        
        has_pronoun = any(pronoun in question for pronoun in self.PRONOUNS)
        
        ref_score = 0.0
        for ref in expected_refs:
            if ref in answer:
                ref_score += 1.0
        
        if expected_refs:
            ref_score = ref_score / len(expected_refs)
        
        if has_pronoun and ref_score > 0:
            ref_score = min(1.0, ref_score + 0.2)
        
        return ref_score
    
    def _calculate_coherence(
        self,
        question: str,
        answer: str,
        history: List[Dict]
    ) -> float:
        """计算上下文连贯性分数"""
        if not history:
            return 1.0
        
        score = 0.5
        
        if len(answer) > 20:
            score += 0.2
        
        if any(pronoun in question for pronoun in self.PRONOUNS):
            if history:
                last_turn = history[-1]
                last_keywords = self._extract_keywords(last_turn.get('answer', ''))
                current_keywords = self._extract_keywords(answer)
                
                overlap = len(set(last_keywords) & set(current_keywords))
                if overlap > 0:
                    score += min(0.3, overlap * 0.1)
        
        return min(1.0, score)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取文本关键词"""
        stopwords = {'的', '是', '在', '有', '和', '了', '我', '你', '他', '这', '那', '就', '也', '都', '还', '要', '会', '能', '对', '与', '或', '但', '如', '而'}
        
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text)
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        
        return list(set(keywords))[:10]
    
    def _check_pronoun_resolution(
        self,
        question: str,
        answer: str,
        history: List[Dict]
    ) -> bool:
        """检查指代消解是否成功"""
        if not history:
            return True
        
        pronoun_in_question = any(pronoun in question for pronoun in self.PRONOUNS)
        
        if not pronoun_in_question:
            return True
        
        last_turn = history[-1]
        last_answer = last_turn.get('answer', '')
        last_keywords = self._extract_keywords(last_answer)
        
        answer_keywords = self._extract_keywords(answer)
        
        overlap = len(set(last_keywords) & set(answer_keywords))
        
        return overlap > 0
    
    def run_scenario(self, scenario: TestScenario) -> ScenarioEvaluationResult:
        """运行单个测试场景"""
        logger.info(f"运行场景: {scenario.scenario_name}")
        
        session_id = self.rag_service.create_conversation_session(
            title=f"测试场景: {scenario.scenario_name}"
        )
        
        conversation_history = []
        turn_results = []
        
        for turn in scenario.turns:
            result = self.rag_service.query(
                question=turn.question,
                session_id=session_id
            )
            
            answer = result.get('answer', '')
            
            turn_result = self.evaluate_turn(turn, answer, conversation_history)
            turn_results.append(turn_result)
            
            conversation_history.append({
                'question': turn.question,
                'answer': answer
            })
            
            logger.info(f"  轮次 {turn.turn_id}: 关键词匹配={turn_result.keyword_match_score:.2f}, "
                       f"上下文引用={turn_result.context_ref_score:.2f}, "
                       f"连贯性={turn_result.coherence_score:.2f}")
        
        avg_coherence = sum(t.coherence_score for t in turn_results) / len(turn_results)
        avg_keyword = sum(t.keyword_match_score for t in turn_results) / len(turn_results)
        avg_context = sum(t.context_ref_score for t in turn_results) / len(turn_results)
        
        coref_turns = [t for t in turn_results if '指代消解' in t.scenario_type]
        coref_accuracy = 0.0
        if coref_turns:
            successful = sum(1 for t in coref_turns if not any('指代' in i for i in t.issues))
            coref_accuracy = successful / len(coref_turns)
        
        context_retention = self._calculate_context_retention(turn_results)
        
        return ScenarioEvaluationResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.scenario_name,
            scenario_type=scenario.scenario_type.value,
            total_turns=len(turn_results),
            avg_coherence_score=avg_coherence,
            avg_keyword_match=avg_keyword,
            avg_context_ref=avg_context,
            coreference_accuracy=coref_accuracy,
            context_retention_score=context_retention,
            turn_results=turn_results
        )
    
    def _calculate_context_retention(self, turn_results: List[TurnEvaluationResult]) -> float:
        """计算上下文保持分数"""
        if len(turn_results) < 2:
            return 1.0
        
        retention_scores = []
        for i in range(1, len(turn_results)):
            current = turn_results[i]
            if current.context_ref_score > 0:
                retention_scores.append(current.context_ref_score)
        
        if not retention_scores:
            return 1.0
        
        return sum(retention_scores) / len(retention_scores)
    
    def run_all_scenarios(self) -> MultiTurnEvaluationReport:
        """运行所有测试场景"""
        if not self.test_scenarios:
            self.load_test_scenarios()
        
        logger.info(f"开始运行 {len(self.test_scenarios)} 个测试场景")
        
        self.evaluation_results = []
        
        for i, scenario in enumerate(self.test_scenarios):
            logger.info(f"\n[{i+1}/{len(self.test_scenarios)}] 运行场景: {scenario.scenario_name}")
            result = self.run_scenario(scenario)
            self.evaluation_results.append(result)
        
        total_turns = sum(r.total_turns for r in self.evaluation_results)
        overall_coherence = sum(r.avg_coherence_score for r in self.evaluation_results) / len(self.evaluation_results)
        overall_keyword = sum(r.avg_keyword_match for r in self.evaluation_results) / len(self.evaluation_results)
        overall_context = sum(r.avg_context_ref for r in self.evaluation_results) / len(self.evaluation_results)
        
        coref_results = [r for r in self.evaluation_results if '指代消解' in r.scenario_type]
        overall_coref = 0.0
        if coref_results:
            overall_coref = sum(r.coreference_accuracy for r in coref_results) / len(coref_results)
        
        by_type = {}
        for result in self.evaluation_results:
            stype = result.scenario_type
            if stype not in by_type:
                by_type[stype] = {
                    'count': 0,
                    'avg_coherence': 0.0,
                    'avg_keyword': 0.0,
                    'avg_context_ref': 0.0,
                    'results': []
                }
            by_type[stype]['count'] += 1
            by_type[stype]['results'].append(result)
        
        for stype in by_type:
            results = by_type[stype]['results']
            by_type[stype]['avg_coherence'] = sum(r.avg_coherence_score for r in results) / len(results)
            by_type[stype]['avg_keyword'] = sum(r.avg_keyword_match for r in results) / len(results)
            by_type[stype]['avg_context_ref'] = sum(r.avg_context_ref for r in results) / len(results)
            del by_type[stype]['results']
        
        report = MultiTurnEvaluationReport(
            test_time=datetime.now().isoformat(),
            total_scenarios=len(self.evaluation_results),
            total_turns=total_turns,
            overall_coherence_score=overall_coherence,
            overall_coreference_accuracy=overall_coref,
            overall_context_retention=overall_context,
            overall_keyword_match=overall_keyword,
            by_scenario_type=by_type,
            scenario_results=self.evaluation_results
        )
        
        return report
    
    def generate_report(
        self,
        report: MultiTurnEvaluationReport = None,
        output_path: str = None
    ) -> str:
        """生成评估报告"""
        if report is None:
            report = self.run_all_scenarios()
        
        lines = []
        lines.append("=" * 70)
        lines.append("多轮对话质量评估报告")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append(f"测试时间: {report.test_time}")
        lines.append(f"测试场景数: {report.total_scenarios}")
        lines.append(f"测试轮次总数: {report.total_turns}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("一、整体评估指标")
        lines.append("-" * 70)
        lines.append("")
        
        lines.append("1. 上下文连贯性得分")
        lines.append("-" * 40)
        coherence = report.overall_coherence_score
        bar = "█" * int(coherence * 20)
        lines.append(f"   连贯性: {coherence:.4f} {bar}")
        lines.append("")
        
        lines.append("2. 指代消解准确率")
        lines.append("-" * 40)
        coref = report.overall_coreference_accuracy
        bar = "█" * int(coref * 20)
        lines.append(f"   准确率: {coref:.4f} {bar}")
        lines.append("")
        
        lines.append("3. 多轮信息融合质量")
        lines.append("-" * 40)
        retention = report.overall_context_retention
        bar = "█" * int(retention * 20)
        lines.append(f"   融合质量: {retention:.4f} {bar}")
        lines.append("")
        
        lines.append("4. 关键词匹配率")
        lines.append("-" * 40)
        keyword = report.overall_keyword_match
        bar = "█" * int(keyword * 20)
        lines.append(f"   匹配率: {keyword:.4f} {bar}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("二、按场景类型分析")
        lines.append("-" * 70)
        lines.append("")
        
        for stype, stats in report.by_scenario_type.items():
            lines.append(f"【{stype}】 (共 {stats['count']} 个场景)")
            lines.append(f"   平均连贯性: {stats['avg_coherence']:.4f}")
            lines.append(f"   平均关键词匹配: {stats['avg_keyword']:.4f}")
            lines.append(f"   平均上下文引用: {stats['avg_context_ref']:.4f}")
            lines.append("")
        
        lines.append("-" * 70)
        lines.append("三、各场景详细结果")
        lines.append("-" * 70)
        lines.append("")
        
        for result in report.scenario_results:
            lines.append(f"场景: {result.scenario_name} ({result.scenario_id})")
            lines.append(f"   类型: {result.scenario_type}")
            lines.append(f"   轮次数: {result.total_turns}")
            lines.append(f"   平均连贯性: {result.avg_coherence_score:.4f}")
            lines.append(f"   指代消解准确率: {result.coreference_accuracy:.4f}")
            lines.append(f"   上下文保持: {result.context_retention_score:.4f}")
            
            if result.turn_results:
                lines.append("   各轮次详情:")
                for tr in result.turn_results:
                    status = "✓" if not tr.issues else "✗"
                    lines.append(f"     [{status}] 轮次{tr.turn_id}: 连贯={tr.coherence_score:.2f}, "
                               f"关键词={tr.keyword_match_score:.2f}, 上下文={tr.context_ref_score:.2f}")
                    if tr.issues:
                        for issue in tr.issues:
                            lines.append(f"         问题: {issue}")
            lines.append("")
        
        lines.append("-" * 70)
        lines.append("四、评估总结")
        lines.append("-" * 70)
        lines.append("")
        
        avg_score = (coherence + coref + retention + keyword) / 4
        
        if avg_score >= 0.8:
            rating = "优秀 ★★★★★"
            suggestion = "多轮对话能力优秀，系统能够很好地理解上下文和指代关系"
        elif avg_score >= 0.6:
            rating = "良好 ★★★★☆"
            suggestion = "多轮对话能力良好，建议优化指代消解和上下文关联逻辑"
        elif avg_score >= 0.4:
            rating = "一般 ★★★☆☆"
            suggestion = "多轮对话能力一般，需要增强上下文记忆和指代理解能力"
        else:
            rating = "待改进 ★★☆☆☆"
            suggestion = "多轮对话能力需要改进，建议检查对话历史传递机制"
        
        lines.append(f"综合评级: {rating}")
        lines.append(f"优化建议: {suggestion}")
        lines.append("")
        
        lines.append("=" * 70)
        lines.append("报告结束")
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"报告已保存到: {output_path}")
        
        return report_text


def main():
    """主函数"""
    print("=" * 70)
    print("多轮对话质量评估工具")
    print("=" * 70)
    print()
    
    evaluator = MultiTurnEvaluator(use_mock=True)
    
    print("1. 加载测试场景...")
    scenarios = evaluator.load_test_scenarios()
    print(f"   加载了 {len(scenarios)} 个测试场景")
    print()
    
    print("2. 运行评估...")
    report = evaluator.run_all_scenarios()
    print()
    
    print("3. 生成报告...")
    report_text = evaluator.generate_report(report)
    print(report_text)
    
    output_dir = "./data/multi_turn_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    scenarios_path = os.path.join(output_dir, f"test_scenarios_{timestamp}.json")
    evaluator.save_scenarios_to_file(scenarios_path)
    
    report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
    evaluator.generate_report(report, report_path)
    
    print(f"\n评估完成！结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
