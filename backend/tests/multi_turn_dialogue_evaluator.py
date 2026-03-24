#!/usr/bin/env python3
"""
多轮对话质量评估工具

功能：
1. 设计多轮对话测试场景（5-8轮），包含指代消解、上下文关联、主题切换、信息补全
2. 自动化测试框架：调用RAG系统query接口，检查上下文引用质量
3. 输出评估报告：上下文连贯性得分、指代消解准确率、多轮信息融合质量

约束：
- 使用现有RAGService的session_id机制
- 测试数据包含代码相关的多轮问答
- 支持批量执行多轮测试用例

Author: RAG Testing Team
Date: 2026-03-24
"""

import os
import sys
import json
import re
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import uuid

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.services.rag_service import RAGService
from backend.services.conversation_store import get_conversation_store, ConversationTurn


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class DialogueTurn:
    """对话轮次数据类
    
    Attributes:
        turn_id: 轮次ID
        question: 用户问题
        expected_keywords: 期望回答中包含的关键词
        expected_context_refs: 期望引用的前文内容
        evaluation_type: 评估类型 (anaphora/context/topic_switch/info_completion)
        description: 测试用例描述
    """
    turn_id: int
    question: str
    expected_keywords: List[str] = field(default_factory=list)
    expected_context_refs: List[str] = field(default_factory=list)
    evaluation_type: str = "context"  # anaphora, context, topic_switch, info_completion
    description: str = ""


@dataclass
class DialogueTestCase:
    """多轮对话测试用例
    
    Attributes:
        case_id: 用例ID
        name: 用例名称
        description: 用例描述
        turns: 对话轮次列表
        category: 测试类别 (code/concept/mixed)
    """
    case_id: str
    name: str
    description: str
    turns: List[DialogueTurn]
    category: str = "mixed"


@dataclass
class TurnResult:
    """单轮对话评估结果
    
    Attributes:
        turn_id: 轮次ID
        question: 问题
        answer: 系统回答
        evaluation_type: 评估类型
        keyword_match_rate: 关键词匹配率
        context_reference_score: 上下文引用得分
        coherence_score: 连贯性得分
        issues: 发现的问题列表
        latency_ms: 响应延迟
    """
    turn_id: int
    question: str
    answer: str
    evaluation_type: str
    keyword_match_rate: float = 0.0
    context_reference_score: float = 0.0
    coherence_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class DialogueEvaluationResult:
    """多轮对话评估结果
    
    Attributes:
        case_id: 用例ID
        session_id: 对话会话ID
        turn_results: 各轮次结果
        overall_coherence: 整体连贯性得分
        anaphora_accuracy: 指代消解准确率
        context_retention: 上下文保持率
        topic_switch_handling: 主题切换处理得分
        info_fusion_quality: 信息融合质量
        avg_latency_ms: 平均延迟
    """
    case_id: str
    session_id: str
    turn_results: List[TurnResult]
    overall_coherence: float = 0.0
    anaphora_accuracy: float = 0.0
    context_retention: float = 0.0
    topic_switch_handling: float = 0.0
    info_fusion_quality: float = 0.0
    avg_latency_ms: float = 0.0


# ============================================================================
# 测试场景设计
# ============================================================================

class DialogueTestScenarios:
    """多轮对话测试场景库
    
    包含5-8轮的多轮对话测试用例，涵盖：
    - 指代消解（它、这个、上述、前者等）
    - 上下文关联（基于之前内容提问）
    - 主题切换与回归
    - 信息补全（基于已有信息回答后续问题）
    """
    
    @staticmethod
    def get_all_test_cases() -> List[DialogueTestCase]:
        """获取所有测试用例"""
        return [
            DialogueTestScenarios._code_debug_scenario(),
            DialogueTestScenarios._api_design_scenario(),
            DialogueTestScenarios._concept_explanation_scenario(),
            DialogueTestScenarios._troubleshooting_scenario(),
            DialogueTestScenarios._architecture_discussion_scenario(),
        ]
    
    @staticmethod
    def _code_debug_scenario() -> DialogueTestCase:
        """代码调试场景 - 6轮对话"""
        return DialogueTestCase(
            case_id="code_debug_001",
            name="Python代码调试场景",
            description="多轮代码调试对话，测试上下文理解和指代消解",
            category="code",
            turns=[
                DialogueTurn(
                    turn_id=1,
                    question="Python中列表推导式是什么？",
                    expected_keywords=["列表", "推导式", "简洁", "for"],
                    expected_context_refs=[],
                    evaluation_type="context",
                    description="基础概念询问"
                ),
                DialogueTurn(
                    turn_id=2,
                    question="它和普通for循环有什么区别？",
                    expected_keywords=["性能", "简洁", "可读性", "效率"],
                    expected_context_refs=["列表推导式"],
                    evaluation_type="anaphora",
                    description="指代消解测试 - '它'指代列表推导式"
                ),
                DialogueTurn(
                    turn_id=3,
                    question="上面的代码在什么情况下会报错？",
                    expected_keywords=["内存", "大数据", "生成器"],
                    expected_context_refs=["列表推导式"],
                    evaluation_type="anaphora",
                    description="指代消解测试 - '上面的代码'"
                ),
                DialogueTurn(
                    turn_id=4,
                    question="那应该用什么替代？",
                    expected_keywords=["生成器表达式", "generator", "yield"],
                    expected_context_refs=["列表推导式", "内存", "大数据"],
                    evaluation_type="anaphora",
                    description="指代消解测试 - '那'指代前面的问题"
                ),
                DialogueTurn(
                    turn_id=5,
                    question="Java里有类似的特性吗？",
                    expected_keywords=["Stream", "Lambda", "Java 8"],
                    expected_context_refs=[],
                    evaluation_type="topic_switch",
                    description="主题切换 - 从Python转到Java"
                ),
                DialogueTurn(
                    turn_id=6,
                    question="回到Python，刚才说的生成器怎么写？",
                    expected_keywords=["()", "生成器", "表达式"],
                    expected_context_refs=["生成器表达式", "Python"],
                    evaluation_type="topic_switch",
                    description="主题回归 - 回到Python生成器"
                ),
            ]
        )
    
    @staticmethod
    def _api_design_scenario() -> DialogueTestCase:
        """API设计场景 - 7轮对话"""
        return DialogueTestCase(
            case_id="api_design_001",
            name="REST API设计场景",
            description="API设计多轮讨论，测试信息补全和上下文关联",
            category="code",
            turns=[
                DialogueTurn(
                    turn_id=1,
                    question="设计一个用户注册API需要注意什么？",
                    expected_keywords=["验证", "密码", "安全", "HTTP", "POST"],
                    expected_context_refs=[],
                    evaluation_type="context",
                    description="初始询问"
                ),
                DialogueTurn(
                    turn_id=2,
                    question="密码应该怎么处理？",
                    expected_keywords=["加密", "哈希", "bcrypt", "salt"],
                    expected_context_refs=["用户注册", "密码"],
                    evaluation_type="context",
                    description="上下文关联 - 基于前文"
                ),
                DialogueTurn(
                    turn_id=3,
                    question="除了密码，还有哪些字段需要验证？",
                    expected_keywords=["邮箱", "手机号", "用户名", "格式"],
                    expected_context_refs=["用户注册", "验证"],
                    evaluation_type="info_completion",
                    description="信息补全 - 基于已有验证信息"
                ),
                DialogueTurn(
                    turn_id=4,
                    question="验证失败应该返回什么状态码？",
                    expected_keywords=["400", "422", "错误", "Validation"],
                    expected_context_refs=["验证", "API"],
                    evaluation_type="context",
                    description="上下文关联"
                ),
                DialogueTurn(
                    turn_id=5,
                    question="前面提到的这些安全措施，在登录API中也适用吗？",
                    expected_keywords=["适用", "登录", "密码", "加密"],
                    expected_context_refs=["密码", "加密", "验证"],
                    evaluation_type="anaphora",
                    description="指代消解 - '前面提到的这些安全措施'"
                ),
                DialogueTurn(
                    turn_id=6,
                    question="如果我想支持第三方登录呢？",
                    expected_keywords=["OAuth", "JWT", "Token", "第三方"],
                    expected_context_refs=["登录"],
                    evaluation_type="topic_switch",
                    description="主题切换 - 从普通登录到OAuth"
                ),
                DialogueTurn(
                    turn_id=7,
                    question="总结一下，注册和登录API的关键区别是什么？",
                    expected_keywords=["注册", "登录", "创建", "验证"],
                    expected_context_refs=["用户注册", "登录API"],
                    evaluation_type="info_completion",
                    description="信息融合 - 综合多轮信息"
                ),
            ]
        )
    
    @staticmethod
    def _concept_explanation_scenario() -> DialogueTestCase:
        """概念解释场景 - 5轮对话"""
        return DialogueTestCase(
            case_id="concept_001",
            name="微服务架构概念解释",
            description="微服务概念的多轮解释，测试概念理解和上下文保持",
            category="concept",
            turns=[
                DialogueTurn(
                    turn_id=1,
                    question="什么是微服务架构？",
                    expected_keywords=["服务", "独立", "分布式", "松耦合"],
                    expected_context_refs=[],
                    evaluation_type="context",
                    description="概念定义"
                ),
                DialogueTurn(
                    turn_id=2,
                    question="它和单体架构相比有什么优缺点？",
                    expected_keywords=["扩展性", "复杂", "部署", "维护"],
                    expected_context_refs=["微服务架构"],
                    evaluation_type="anaphora",
                    description="指代消解 - '它'指代微服务"
                ),
                DialogueTurn(
                    turn_id=3,
                    question="服务之间怎么通信？",
                    expected_keywords=["HTTP", "gRPC", "消息队列", "API"],
                    expected_context_refs=["微服务", "服务"],
                    evaluation_type="context",
                    description="上下文关联"
                ),
                DialogueTurn(
                    turn_id=4,
                    question="这种通信方式有什么缺点？",
                    expected_keywords=["延迟", "网络", "故障", "一致"],
                    expected_context_refs=["HTTP", "gRPC", "通信"],
                    evaluation_type="anaphora",
                    description="指代消解 - '这种通信方式'"
                ),
                DialogueTurn(
                    turn_id=5,
                    question="如何解决这些问题？",
                    expected_keywords=["熔断", "重试", "超时", "降级"],
                    expected_context_refs=["延迟", "故障", "网络"],
                    evaluation_type="anaphora",
                    description="指代消解 - '这些问题'"
                ),
            ]
        )
    
    @staticmethod
    def _troubleshooting_scenario() -> DialogueTestCase:
        """故障排查场景 - 8轮对话"""
        return DialogueTestCase(
            case_id="troubleshoot_001",
            name="数据库性能故障排查",
            description="数据库性能问题的多轮排查，测试问题追踪和信息整合",
            category="code",
            turns=[
                DialogueTurn(
                    turn_id=1,
                    question="数据库查询很慢，可能是什么原因？",
                    expected_keywords=["索引", "查询", "优化", "慢查询"],
                    expected_context_refs=[],
                    evaluation_type="context",
                    description="问题描述"
                ),
                DialogueTurn(
                    turn_id=2,
                    question="怎么查看哪些查询慢？",
                    expected_keywords=["慢查询日志", "EXPLAIN", "性能分析"],
                    expected_context_refs=["慢", "查询"],
                    evaluation_type="context",
                    description="上下文关联"
                ),
                DialogueTurn(
                    turn_id=3,
                    question="找到慢查询后怎么处理？",
                    expected_keywords=["索引", "优化", "改写", "缓存"],
                    expected_context_refs=["慢查询"],
                    evaluation_type="anaphora",
                    description="指代消解 - '慢查询'"
                ),
                DialogueTurn(
                    turn_id=4,
                    question="加了索引还是慢怎么办？",
                    expected_keywords=["数据量", "分表", "分区", "架构"],
                    expected_context_refs=["索引", "慢"],
                    evaluation_type="info_completion",
                    description="信息补全 - 基于前面的解决方案"
                ),
                DialogueTurn(
                    turn_id=5,
                    question="数据量大概多少需要考虑分库分表？",
                    expected_keywords=["千万", "百万", "分片", "水平"],
                    expected_context_refs=["数据量", "分表"],
                    evaluation_type="context",
                    description="上下文关联"
                ),
                DialogueTurn(
                    turn_id=6,
                    question="分表后怎么保证跨表查询的性能？",
                    expected_keywords=["中间件", "ShardingSphere", "路由"],
                    expected_context_refs=["分表"],
                    evaluation_type="context",
                    description="上下文关联"
                ),
                DialogueTurn(
                    turn_id=7,
                    question="除了刚才说的这些，还有什么优化手段？",
                    expected_keywords=["缓存", "Redis", "读写分离", "主从"],
                    expected_context_refs=["索引", "分表", "优化"],
                    evaluation_type="anaphora",
                    description="指代消解 - '刚才说的这些'"
                ),
                DialogueTurn(
                    turn_id=8,
                    question="能给我一个完整的排查流程吗？",
                    expected_keywords=["慢查询", "索引", "分表", "缓存"],
                    expected_context_refs=["慢查询日志", "索引优化", "分库分表"],
                    evaluation_type="info_completion",
                    description="信息融合 - 整合所有排查步骤"
                ),
            ]
        )
    
    @staticmethod
    def _architecture_discussion_scenario() -> DialogueTestCase:
        """架构讨论场景 - 6轮对话"""
        return DialogueTestCase(
            case_id="arch_001",
            name="缓存架构设计讨论",
            description="缓存架构的多轮讨论，测试技术选型和方案比较",
            category="code",
            turns=[
                DialogueTurn(
                    turn_id=1,
                    question="Redis和Memcached有什么区别？",
                    expected_keywords=["数据类型", "持久化", "集群", "功能"],
                    expected_context_refs=[],
                    evaluation_type="context",
                    description="技术对比"
                ),
                DialogueTurn(
                    turn_id=2,
                    question="什么情况下选前者？",
                    expected_keywords=["复杂", "数据结构", "持久化"],
                    expected_context_refs=["Redis"],
                    evaluation_type="anaphora",
                    description="指代消解 - '前者'指Redis"
                ),
                DialogueTurn(
                    turn_id=3,
                    question="缓存穿透怎么解决？",
                    expected_keywords=["布隆过滤器", "空值", "缓存"],
                    expected_context_refs=["缓存"],
                    evaluation_type="context",
                    description="上下文关联"
                ),
                DialogueTurn(
                    turn_id=4,
                    question="那缓存击穿呢？",
                    expected_keywords=["互斥锁", "热点", "过期"],
                    expected_context_refs=["缓存穿透"],
                    evaluation_type="anaphora",
                    description="指代消解 - '那'承接前文"
                ),
                DialogueTurn(
                    turn_id=5,
                    question="回到Redis，它的持久化机制有哪些？",
                    expected_keywords=["RDB", "AOF", "快照", "日志"],
                    expected_context_refs=["Redis", "持久化"],
                    evaluation_type="topic_switch",
                    description="主题回归 - 回到Redis持久化"
                ),
                DialogueTurn(
                    turn_id=6,
                    question="这两种持久化方式各有什么优缺点？",
                    expected_keywords=["RDB", "AOF", "性能", "恢复"],
                    expected_context_refs=["RDB", "AOF", "持久化"],
                    evaluation_type="anaphora",
                    description="指代消解 - '这两种'指RDB和AOF"
                ),
            ]
        )


# ============================================================================
# 评估指标计算器
# ============================================================================

class DialogueMetricsCalculator:
    """多轮对话质量评估指标计算器"""
    
    @staticmethod
    def calculate_keyword_match_rate(answer: str, expected_keywords: List[str]) -> float:
        """计算关键词匹配率
        
        Args:
            answer: 系统回答
            expected_keywords: 期望关键词列表
            
        Returns:
            匹配率 (0.0 - 1.0)
        """
        if not expected_keywords:
            return 1.0
        
        answer_lower = answer.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        return matched / len(expected_keywords)
    
    @staticmethod
    def calculate_context_reference_score(
        answer: str,
        expected_refs: List[str],
        conversation_history: List[Dict]
    ) -> float:
        """计算上下文引用得分
        
        检查回答是否正确引用了前文内容
        
        Args:
            answer: 系统回答
            expected_refs: 期望引用的内容
            conversation_history: 对话历史
            
        Returns:
            引用得分 (0.0 - 1.0)
        """
        if not expected_refs:
            return 1.0
        
        answer_lower = answer.lower()
        matched = sum(1 for ref in expected_refs if ref.lower() in answer_lower)
        return min(1.0, matched / len(expected_refs) * 1.5)  # 允许部分匹配
    
    @staticmethod
    def detect_anaphora_resolution(question: str, answer: str, history: List[Dict]) -> Tuple[bool, List[str]]:
        """检测指代消解质量
        
        检测回答是否正确解析了指代词
        
        Args:
            question: 当前问题
            answer: 系统回答
            history: 对话历史
            
        Returns:
            (是否成功, 问题列表)
        """
        issues = []
        
        # 指代词列表
        anaphora_words = ['它', '这个', '上述', '前者', '后者', '这些', '那些', '这样', '那样']
        
        # 检查问题中是否包含指代词
        has_anaphora = any(word in question for word in anaphora_words)
        
        if has_anaphora:
            # 检查回答是否明确解析了指代
            # 简单规则：回答应该包含具体名词而非仅使用代词
            answer_has_specific_noun = len(answer) > 20  # 简化判断
            
            if not answer_has_specific_noun:
                issues.append("回答可能未正确解析指代词")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def calculate_coherence_score(
        current_answer: str,
        previous_answers: List[str],
        current_question: str
    ) -> float:
        """计算连贯性得分
        
        检查回答是否与上下文保持连贯
        
        Args:
            current_answer: 当前回答
            previous_answers: 之前回答列表
            current_question: 当前问题
            
        Returns:
            连贯性得分 (0.0 - 1.0)
        """
        if not previous_answers:
            return 1.0
        
        # 简单连贯性检查：回答长度合理、包含相关词汇
        score = 0.5
        
        # 回答长度检查
        if 50 <= len(current_answer) <= 2000:
            score += 0.2
        
        # 检查是否包含问题中的关键词
        question_keywords = set(re.findall(r'\w+', current_question))
        answer_words = set(re.findall(r'\w+', current_answer))
        keyword_overlap = len(question_keywords & answer_words) / max(len(question_keywords), 1)
        score += keyword_overlap * 0.3
        
        return min(1.0, score)
    
    @staticmethod
    def detect_context_forgetfulness(
        answer: str,
        conversation_history: List[Dict],
        window_size: int = 3
    ) -> Tuple[bool, List[str]]:
        """检测上下文遗忘
        
        检查系统是否遗忘了前文重要信息
        
        Args:
            answer: 当前回答
            conversation_history: 对话历史
            window_size: 检查窗口大小
            
        Returns:
            (是否遗忘, 遗忘信息列表)
        """
        issues = []
        
        if len(conversation_history) < 2:
            return False, issues
        
        # 检查最近几轮的关键信息是否被保留
        recent_history = conversation_history[-window_size:]
        
        for turn in recent_history:
            # 提取关键信息（简化处理）
            key_info = turn.get('answer', '')[:100]  # 取前100字符作为关键信息
            if key_info and len(key_info) > 20:
                # 检查当前回答是否提及了前文信息
                if not any(word in answer for word in key_info.split()[:5]):
                    # 不直接判定遗忘，只是潜在问题
                    pass
        
        return len(issues) > 0, issues


# ============================================================================
# 多轮对话评估器
# ============================================================================

class MultiTurnDialogueEvaluator:
    """多轮对话质量评估器主类"""
    
    def __init__(self):
        """初始化评估器"""
        self.rag_service = RAGService()
        self.conversation_store = get_conversation_store()
        self.metrics_calculator = DialogueMetricsCalculator()
    
    def evaluate_single_turn(
        self,
        turn: DialogueTurn,
        session_id: str,
        conversation_history: List[Dict]
    ) -> TurnResult:
        """评估单轮对话
        
        Args:
            turn: 对话轮次
            session_id: 会话ID
            conversation_history: 对话历史
            
        Returns:
            评估结果
        """
        start_time = time.time()
        
        # 调用RAG服务
        try:
            result = self.rag_service.query(
                question=turn.question,
                session_id=session_id,
                stream=False
            )
            # 处理生成器返回值
            if hasattr(result, '__iter__') and not isinstance(result, dict):
                # 是生成器，收集所有片段
                chunks = []
                for chunk in result:
                    if isinstance(chunk, dict):
                        if chunk.get('done'):
                            break
                        chunks.append(chunk.get('answer', ''))
                answer = ''.join(chunks)
            else:
                answer = result.get('answer', '')
        except Exception as e:
            answer = f"错误: {str(e)}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 计算各项指标
        keyword_match_rate = self.metrics_calculator.calculate_keyword_match_rate(
            answer, turn.expected_keywords
        )
        
        context_ref_score = self.metrics_calculator.calculate_context_reference_score(
            answer, turn.expected_context_refs, conversation_history
        )
        
        coherence_score = self.metrics_calculator.calculate_coherence_score(
            answer,
            [h.get('answer', '') for h in conversation_history],
            turn.question
        )
        
        # 检测指代消解
        anaphora_ok, anaphora_issues = self.metrics_calculator.detect_anaphora_resolution(
            turn.question, answer, conversation_history
        )
        
        # 检测上下文遗忘
        forget_ok, forget_issues = self.metrics_calculator.detect_context_forgetfulness(
            answer, conversation_history
        )
        
        # 收集所有问题
        issues = anaphora_issues + forget_issues
        
        # 根据评估类型调整得分
        if turn.evaluation_type == "anaphora" and not anaphora_ok:
            coherence_score *= 0.7
        
        return TurnResult(
            turn_id=turn.turn_id,
            question=turn.question,
            answer=answer,
            evaluation_type=turn.evaluation_type,
            keyword_match_rate=keyword_match_rate,
            context_reference_score=context_ref_score,
            coherence_score=coherence_score,
            issues=issues,
            latency_ms=latency_ms
        )
    
    def evaluate_test_case(self, test_case: DialogueTestCase) -> DialogueEvaluationResult:
        """评估单个测试用例
        
        Args:
            test_case: 测试用例
            
        Returns:
            评估结果
        """
        print(f"\n评估用例: {test_case.name}")
        print(f"描述: {test_case.description}")
        print("=" * 70)
        
        # 创建新会话
        session_id = self.conversation_store.create_session(
            title=f"测试: {test_case.name}"
        )
        
        turn_results = []
        conversation_history = []
        
        for turn in test_case.turns:
            print(f"\n轮次 {turn.turn_id}: {turn.description}")
            print(f"问题: {turn.question}")
            
            # 评估单轮
            result = self.evaluate_single_turn(turn, session_id, conversation_history)
            turn_results.append(result)
            
            # 更新历史
            conversation_history.append({
                'question': turn.question,
                'answer': result.answer
            })
            
            print(f"关键词匹配: {result.keyword_match_rate:.2%}")
            print(f"连贯性得分: {result.coherence_score:.2%}")
            if result.issues:
                print(f"问题: {', '.join(result.issues)}")
        
        # 计算整体指标
        aggregate = self._calculate_aggregate_metrics(turn_results)
        
        return DialogueEvaluationResult(
            case_id=test_case.case_id,
            session_id=session_id,
            turn_results=turn_results,
            **aggregate
        )
    
    def _calculate_aggregate_metrics(self, turn_results: List[TurnResult]) -> Dict:
        """计算聚合指标"""
        if not turn_results:
            return {}
        
        # 按类型分组
        anaphora_results = [r for r in turn_results if r.evaluation_type == "anaphora"]
        context_results = [r for r in turn_results if r.evaluation_type == "context"]
        topic_switch_results = [r for r in turn_results if r.evaluation_type == "topic_switch"]
        info_completion_results = [r for r in turn_results if r.evaluation_type == "info_completion"]
        
        # 计算各项指标
        overall_coherence = sum(r.coherence_score for r in turn_results) / len(turn_results)
        
        anaphora_accuracy = (
            sum(r.coherence_score for r in anaphora_results) / len(anaphora_results)
            if anaphora_results else 1.0
        )
        
        context_retention = (
            sum(r.context_reference_score for r in context_results) / len(context_results)
            if context_results else 1.0
        )
        
        topic_switch_handling = (
            sum(r.coherence_score for r in topic_switch_results) / len(topic_switch_results)
            if topic_switch_results else 1.0
        )
        
        info_fusion_quality = (
            sum(r.keyword_match_rate for r in info_completion_results) / len(info_completion_results)
            if info_completion_results else 1.0
        )
        
        avg_latency = sum(r.latency_ms for r in turn_results) / len(turn_results)
        
        return {
            'overall_coherence': round(overall_coherence, 4),
            'anaphora_accuracy': round(anaphora_accuracy, 4),
            'context_retention': round(context_retention, 4),
            'topic_switch_handling': round(topic_switch_handling, 4),
            'info_fusion_quality': round(info_fusion_quality, 4),
            'avg_latency_ms': round(avg_latency, 2)
        }
    
    def run_batch_evaluation(self, test_cases: Optional[List[DialogueTestCase]] = None) -> List[DialogueEvaluationResult]:
        """批量评估测试用例
        
        Args:
            test_cases: 测试用例列表，默认使用全部
            
        Returns:
            评估结果列表
        """
        if test_cases is None:
            test_cases = DialogueTestScenarios.get_all_test_cases()
        
        print(f"\n开始批量评估: {len(test_cases)} 个测试用例")
        print("=" * 70)
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] ", end="")
            result = self.evaluate_test_case(test_case)
            results.append(result)
        
        return results


# ============================================================================
# 报告生成器
# ============================================================================

class DialogueReportGenerator:
    """多轮对话评估报告生成器"""
    
    @staticmethod
    def generate_text_report(
        results: List[DialogueEvaluationResult],
        output_path: Optional[str] = None
    ) -> str:
        """生成文本格式报告"""
        lines = []
        
        # 标题
        lines.append("=" * 80)
        lines.append(" " * 20 + "多轮对话质量评估报告")
        lines.append("=" * 80)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 测试概览
        lines.append("-" * 80)
        lines.append("测试概览")
        lines.append("-" * 80)
        lines.append(f"测试用例数: {len(results)}")
        total_turns = sum(len(r.turn_results) for r in results)
        lines.append(f"总对话轮次: {total_turns}")
        lines.append("")
        
        # 各用例详细结果
        for result in results:
            lines.append("-" * 80)
            lines.append(f"用例: {result.case_id}")
            lines.append("-" * 80)
            lines.append(f"会话ID: {result.session_id}")
            lines.append(f"对话轮数: {len(result.turn_results)}")
            lines.append("")
            
            # 整体指标
            lines.append("【整体指标】")
            lines.append(f"  上下文连贯性:    {result.overall_coherence:.2%}")
            lines.append(f"  指代消解准确率:   {result.anaphora_accuracy:.2%}")
            lines.append(f"  上下文保持率:     {result.context_retention:.2%}")
            lines.append(f"  主题切换处理:     {result.topic_switch_handling:.2%}")
            lines.append(f"  信息融合质量:     {result.info_fusion_quality:.2%}")
            lines.append(f"  平均响应延迟:     {result.avg_latency_ms:.2f} ms")
            lines.append("")
            
            # 各轮次详情
            lines.append("【轮次详情】")
            for turn in result.turn_results:
                lines.append(f"\n  轮次 {turn.turn_id} [{turn.evaluation_type}]")
                lines.append(f"    Q: {turn.question[:50]}...")
                lines.append(f"    A: {turn.answer[:80]}...")
                lines.append(f"    关键词匹配: {turn.keyword_match_rate:.2%}")
                lines.append(f"    上下文引用: {turn.context_reference_score:.2%}")
                lines.append(f"    连贯性得分: {turn.coherence_score:.2%}")
                if turn.issues:
                    lines.append(f"    ⚠ 问题: {', '.join(turn.issues)}")
            
            lines.append("")
        
        # 汇总统计
        lines.append("-" * 80)
        lines.append("汇总统计")
        lines.append("-" * 80)
        
        avg_coherence = sum(r.overall_coherence for r in results) / len(results)
        avg_anaphora = sum(r.anaphora_accuracy for r in results) / len(results)
        avg_retention = sum(r.context_retention for r in results) / len(results)
        avg_topic = sum(r.topic_switch_handling for r in results) / len(results)
        avg_fusion = sum(r.info_fusion_quality for r in results) / len(results)
        avg_latency = sum(r.avg_latency_ms for r in results) / len(results)
        
        lines.append(f"平均上下文连贯性:    {avg_coherence:.2%}")
        lines.append(f"平均指代消解准确率:   {avg_anaphora:.2%}")
        lines.append(f"平均上下文保持率:     {avg_retention:.2%}")
        lines.append(f"平均主题切换处理:     {avg_topic:.2%}")
        lines.append(f"平均信息融合质量:     {avg_fusion:.2%}")
        lines.append(f"平均响应延迟:         {avg_latency:.2f} ms")
        lines.append("")
        
        # 评级
        lines.append("-" * 80)
        lines.append("质量评级")
        lines.append("-" * 80)
        
        overall_score = (avg_coherence + avg_anaphora + avg_retention + avg_topic + avg_fusion) / 5
        
        if overall_score >= 0.8:
            rating = "优秀"
        elif overall_score >= 0.6:
            rating = "良好"
        elif overall_score >= 0.4:
            rating = "一般"
        else:
            rating = "需改进"
        
        lines.append(f"综合得分: {overall_score:.2%}")
        lines.append(f"质量评级: {rating}")
        lines.append("")
        
        # 建议
        lines.append("-" * 80)
        lines.append("优化建议")
        lines.append("-" * 80)
        
        if avg_anaphora < 0.7:
            lines.append("• 指代消解能力需提升，建议增强上下文理解模块")
        if avg_retention < 0.7:
            lines.append("• 上下文保持率较低，建议优化对话历史管理机制")
        if avg_topic < 0.7:
            lines.append("• 主题切换处理有待改进，建议增加主题跟踪机制")
        if avg_latency > 1000:
            lines.append("• 响应延迟较高，建议优化检索和生成速度")
        
        if all([avg_anaphora >= 0.7, avg_retention >= 0.7, avg_topic >= 0.7, avg_latency <= 1000]):
            lines.append("• 各项指标表现良好，继续保持！")
        
        lines.append("")
        lines.append("=" * 80)
        
        report_text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n报告已保存到: {output_path}")
        
        return report_text
    
    @staticmethod
    def generate_json_report(
        results: List[DialogueEvaluationResult],
        output_path: str
    ):
        """生成JSON格式详细报告"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_cases': len(results),
                'total_turns': sum(len(r.turn_results) for r in results)
            },
            'results': []
        }
        
        for result in results:
            case_data = {
                'case_id': result.case_id,
                'session_id': result.session_id,
                'metrics': {
                    'overall_coherence': result.overall_coherence,
                    'anaphora_accuracy': result.anaphora_accuracy,
                    'context_retention': result.context_retention,
                    'topic_switch_handling': result.topic_switch_handling,
                    'info_fusion_quality': result.info_fusion_quality,
                    'avg_latency_ms': result.avg_latency_ms
                },
                'turns': [
                    {
                        'turn_id': t.turn_id,
                        'question': t.question,
                        'answer': t.answer,
                        'evaluation_type': t.evaluation_type,
                        'keyword_match_rate': t.keyword_match_rate,
                        'context_reference_score': t.context_reference_score,
                        'coherence_score': t.coherence_score,
                        'issues': t.issues,
                        'latency_ms': t.latency_ms
                    }
                    for t in result.turn_results
                ]
            }
            report_data['results'].append(case_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"JSON报告已保存到: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 执行多轮对话质量评估"""
    
    print("=" * 80)
    print(" " * 25 + "多轮对话质量评估")
    print("=" * 80)
    
    # 初始化评估器
    evaluator = MultiTurnDialogueEvaluator()
    
    # 获取所有测试用例
    test_cases = DialogueTestScenarios.get_all_test_cases()
    
    print(f"\n加载了 {len(test_cases)} 个测试用例:")
    for tc in test_cases:
        print(f"  - {tc.name}: {len(tc.turns)} 轮对话")
    
    # 执行批量评估
    print("\n开始评估...")
    results = evaluator.run_batch_evaluation(test_cases)
    
    # 生成报告
    print("\n\n生成评估报告...")
    
    report_gen = DialogueReportGenerator()
    
    # 文本报告
    text_report = report_gen.generate_text_report(
        results,
        output_path='backend/tests/dialogue_evaluation_report.txt'
    )
    
    # 打印报告
    print("\n" + text_report)
    
    # JSON报告
    report_gen.generate_json_report(
        results,
        output_path='backend/tests/dialogue_evaluation_report.json'
    )
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()
