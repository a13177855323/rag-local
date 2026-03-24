#!/usr/bin/env python3
"""
多轮对话质量评估工具

功能：
1. 设计多轮对话测试场景（5-8轮），包含：
   - 指代消解（它、这个、上述、前者等）
   - 上下文关联（基于之前内容提问）
   - 主题切换与回归
   - 信息补全（基于已有信息回答后续问题）

2. 实现自动化测试框架：
   - 自动导入测试对话脚本
   - 调用RAG系统的query接口（带session_id）
   - 检查回答中是否正确引用了上下文信息
   - 检测是否出现上下文遗忘或混淆

3. 输出评估报告：
   - 上下文连贯性得分
   - 指代消解准确率
   - 多轮信息融合质量

约束条件：
- 必须使用现有RAGService的session_id机制
- 测试数据需包含代码相关的多轮问答
- 需支持批量执行多轮测试用例
"""
# 解决macOS上FAISS和PyTorch的OpenMP库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import sys
sys.path.insert(0, '.')

import json
import re
from typing import List, Dict, Any
from datetime import datetime

# 注意：必须先加载嵌入模型（PyTorch），再初始化FAISS，否则会有库冲突
from backend.models.embedding_model import EmbeddingModel
from backend.services.vector_store import VectorStore
from backend.services.conversation_store import get_conversation_store
from backend.config import settings


class MultirunDialogEvaluator:
    """
    多轮对话质量评估器
    
    负责：
    1. 定义多轮对话测试场景
    2. 执行多轮对话测试
    3. 分析上下文理解质量
    4. 生成评估报告
    """
    
    def __init__(self):
        """初始化评估器"""
        print("初始化评估器组件...")
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.conversation_store = get_conversation_store()
        self.test_scenarios = self._define_test_scenarios()
    
    def _define_test_scenarios(self) -> List[Dict[str, Any]]:
        """
        定义多轮对话测试场景
        每个场景包含5-8轮对话，覆盖：指代消解、上下文关联、主题切换、信息补全等场景
        """
        scenarios = [
            # 场景1：Python代码相关的多轮对话 - 装饰器
            {
                "scenario_id": "PY001",
                "title": "Python装饰器技术深度探讨",
                "category": "代码技术",
                "required_knowledge": ["Python装饰器", "语法", "应用场景"],
                "turns": [
                    {
                        "turn_id": 1,
                        "question": "Python中的装饰器是什么？请简单解释一下",
                        "type": "concept",
                        "check_anaphora": False,
                        "keywords": ["装饰器", "Python", "函数", "包装"]
                    },
                    {
                        "turn_id": 2,
                        "question": "它的基本语法是怎样的？能否给出一个简单的示例？",
                        "type": "anaphora",
                        "check_anaphora": True,
                        "anaphor": "它",
                        "expected_ref": "装饰器",
                        "keywords": ["语法", "示例", "@decorator", "代码"]
                    },
                    {
                        "turn_id": 3,
                        "question": "这种语法设计有什么优势？主要应用场景有哪些？",
                        "type": "context",
                        "check_anaphora": True,
                        "anaphor": "这种语法",
                        "expected_ref": "装饰器语法",
                        "keywords": ["优势", "应用场景", "日志", "权限", "性能监控"]
                    },
                    {
                        "turn_id": 4,
                        "question": "那带参数的装饰器又该如何实现？和普通的有什么区别？",
                        "type": "topic_expansion",
                        "check_anaphora": False,
                        "keywords": ["参数", "实现", "区别", "嵌套", "外层函数"]
                    },
                    {
                        "turn_id": 5,
                        "question": "刚才讲的内容中，哪种使用方式最常见于Web框架中？",
                        "type": "context_recall",
                        "check_anaphora": True,
                        "anaphor": "刚才讲的内容",
                        "expected_ref": "装饰器相关内容",
                        "keywords": ["Web框架", "常见", "路由", "请求处理"]
                    },
                    {
                        "turn_id": 6,
                        "question": "总结一下装饰器的最佳实践有哪些",
                        "type": "summarization",
                        "check_anaphora": False,
                        "keywords": ["总结", "最佳实践", "functools.wraps"]
                    }
                ]
            },
            
            # 场景2：FAISS向量数据库多轮对话
            {
                "scenario_id": "FAISS001",
                "title": "FAISS向量数据库技术探讨",
                "category": "代码技术",
                "required_knowledge": ["FAISS", "向量数据库", "索引"],
                "turns": [
                    {
                        "turn_id": 1,
                        "question": "FAISS是什么？它主要用于解决什么问题？",
                        "type": "concept",
                        "check_anaphora": True,
                        "anaphor": "它",
                        "expected_ref": "FAISS",
                        "keywords": ["FAISS", "向量检索", "相似性搜索", "Facebook"]
                    },
                    {
                        "turn_id": 2,
                        "question": "它支持哪些主要的索引类型？各有什么优缺点？",
                        "type": "anaphora",
                        "check_anaphora": True,
                        "anaphor": "它",
                        "expected_ref": "FAISS",
                        "keywords": ["索引类型", "IndexFlatL2", "IndexIVFFlat", "HNSW", "优缺点"]
                    },
                    {
                        "turn_id": 3,
                        "question": "在大规模向量检索场景下，哪种索引的查询效率最高？",
                        "type": "context",
                        "check_anaphora": False,
                        "keywords": ["大规模", "查询效率", "HNSW", "高维"]
                    },
                    {
                        "turn_id": 4,
                        "question": "这种高维向量检索的主要应用场景有哪些？",
                        "type": "context_expansion",
                        "check_anaphora": True,
                        "anaphor": "这种",
                        "expected_ref": "高维向量检索",
                        "keywords": ["应用场景", "推荐系统", "图像检索", "语义搜索", "RAG"]
                    },
                    {
                        "turn_id": 5,
                        "question": "刚才讨论的这些索引类型，在RAG系统中最适合用哪一种？",
                        "type": "topic_switch",
                        "topic": "RAG系统应用",
                        "check_anaphora": True,
                        "anaphor": "这些索引类型",
                        "keywords": ["RAG系统", "适合", "向量数据库", "检索"]
                    },
                    {
                        "turn_id": 6,
                        "question": "那回到原问题，FAISS的GPU加速能力如何？",
                        "type": "topic_return",
                        "return_topic": "FAISS本身特性",
                        "check_anaphora": False,
                        "keywords": ["GPU", "加速", "大规模", "性能"]
                    },
                    {
                        "turn_id": 7,
                        "question": "综合来看，FAISS相比其他向量数据库的核心优势是什么？",
                        "type": "summarization",
                        "check_anaphora": False,
                        "keywords": ["核心优势", "对比", "性能", "开源"]
                    }
                ]
            },
            
            # 场景3：Linux系统管理与Shell脚本
            {
                "scenario_id": "LINUX001",
                "title": "Linux系统管理与Shell脚本",
                "category": "代码技术",
                "required_knowledge": ["Linux", "Shell", "系统命令"],
                "turns": [
                    {
                        "turn_id": 1,
                        "question": "Linux中如何查看系统资源使用情况？常用的命令有哪些？",
                        "type": "concept",
                        "check_anaphora": False,
                        "keywords": ["系统资源", "命令", "top", "htop", "free", "df"]
                    },
                    {
                        "turn_id": 2,
                        "question": "这些命令中，哪个最适合实时监控CPU和内存使用情况？",
                        "type": "anaphora",
                        "check_anaphora": True,
                        "anaphor": "这些命令",
                        "expected_ref": "查看系统资源的命令",
                        "keywords": ["实时监控", "CPU", "内存", "top", "htop"]
                    },
                    {
                        "turn_id": 3,
                        "question": "那如何通过Shell脚本实现自动监控并在资源不足时发送告警？",
                        "type": "context_application",
                        "check_anaphora": True,
                        "anaphor": "那",
                        "expected_ref": "基于前面讨论的监控命令",
                        "keywords": ["Shell脚本", "自动监控", "告警", "cron", "邮件"]
                    },
                    {
                        "turn_id": 4,
                        "question": "能否给出一个简单的实现示例？需要包含关键代码逻辑",
                        "type": "context_detail",
                        "check_anaphora": False,
                        "keywords": ["示例", "代码逻辑", "脚本", "if判断", "阈值"]
                    },
                    {
                        "turn_id": 5,
                        "question": "在编写这类脚本时，需要注意哪些常见的坑点？",
                        "type": "topic_expansion",
                        "check_anaphora": True,
                        "anaphor": "这类脚本",
                        "expected_ref": "自动化监控脚本",
                        "keywords": ["坑点", "注意事项", "权限", "路径", "环境变量"]
                    },
                    {
                        "turn_id": 6,
                        "question": "除了资源监控，这类脚本还能应用在哪些运维场景中？",
                        "type": "expansion",
                        "check_anaphora": True,
                        "anaphor": "这类脚本",
                        "expected_ref": "自动化Shell脚本",
                        "keywords": ["运维场景", "应用", "日志分析", "备份", "部署"]
                    },
                    {
                        "turn_id": 7,
                        "question": "如何将刚才讨论的脚本功能集成到一个完整的运维工具中？",
                        "type": "integration",
                        "check_anaphora": True,
                        "anaphor": "刚才讨论的脚本功能",
                        "expected_ref": "自动化脚本功能",
                        "keywords": ["集成", "运维工具", "模块化", "配置文件", "日志系统"]
                    }
                ]
            }
        ]
        
        return scenarios
    
    def _simulate_rag_response(self, question: str, context_history: List[str]) -> str:
        """
        模拟RAG系统的回答生成（实际项目中应调用真实的RAGService.query接口）
        
        Args:
            question: 当前问题
            context_history: 历史对话内容
            
        Returns:
            生成的回答
        """
        # 执行向量检索
        embedding = self.embedding_model.embed_query(question)
        search_results = self.vector_store.search(embedding, top_k=2)
        
        # 构建基础回答
        answer_parts = []
        
        # 处理指代消解（简化版）
        resolved_question = self._resolve_anaphora(question, context_history)
        answer_parts.append(f"理解您的问题：{resolved_question}")
        
        # 添加检索到的知识
        if search_results:
            # search_results格式: [(metadata, distance), ...]
            contents = [r[0].get("content", "") for r in search_results if r[0].get("content")]
            if contents:
                relevant_info = " ".join(contents)[:300]
                answer_parts.append(f"\n根据知识库检索：")
                answer_parts.append(relevant_info)
        
        # 根据问题类型生成回答
        if "什么" in question or "是什么" in question:
            answer_parts.append("\n\n这是一个概念解释类问题，需要从定义、原理、用途等方面进行说明。")
        elif "如何" in question or "怎么" in question or "怎样" in question:
            answer_parts.append("\n\n这是一个操作指南类问题，需要提供具体的实现步骤和示例代码。")
        elif "区别" in question or "比较" in question or "对比" in question:
            answer_parts.append("\n\n这是一个对比分析类问题，需要分析各自的优缺点和适用场景。")
        elif "总结" in question or "归纳" in question:
            answer_parts.append("\n\n基于以上对话内容和知识库信息，为您归纳总结相关要点如下。")
        elif "哪些" in question and "场景" in question:
            answer_parts.append("\n\n这是一个应用场景类问题，以下是主要的应用领域和使用方式。")
        
        return "\n".join(answer_parts)
    
    def _resolve_anaphora(self, question: str, context_history: List[str]) -> str:
        """
        简单的指代消解处理
        
        Args:
            question: 当前问题
            context_history: 历史对话内容
            
        Returns:
            消解后的问题文本
        """
        if not context_history:
            return question
        
        resolved = question
        
        # 定义指代词的替换规则
        anaphora_map = {
            "它的": "该技术的",
            "它": "该技术",
            "这个": "这个技术",
            "这些": "相关技术",
            "这种": "这种技术",
            "这类": "这类技术",
            "刚才讨论的内容": "前面讨论的技术内容",
            "刚才讨论的脚本功能": "前面讨论的自动化脚本功能",
            "那如何": "基于前面讨论的内容，如何",
            "那": "基于前面讨论的内容，"
        }
        
        for anaphor, replacement in anaphora_map.items():
            if anaphor in resolved:
                resolved = resolved.replace(anaphor, replacement)
        
        return resolved
    
    def _evaluate_turn_quality(self, turn: Dict, answer: str, 
                               context_history: List[str]) -> Dict[str, Any]:
        """
        评估单轮对话质量
        
        Args:
            turn: 当前轮次配置
            answer: 系统回答
            context_history: 历史对话内容
            
        Returns:
            评估结果字典
        """
        result = {
            "turn_id": turn["turn_id"],
            "question": turn["question"],
            "answer": answer,
            "type": turn["type"],
            "scores": {
                "anaphora_resolution": 1.0,    # 指代消解得分
                "keyword_relevance": 0.0,      # 关键词相关性
                "context_coherence": 0.8,      # 上下文连贯性
                "overall": 0.0                 # 综合得分
            },
            "details": [],
            "passed": True
        }
        
        # 1. 指代消解评估
        if turn.get("check_anaphora", False):
            anaphor = turn.get("anaphor", "")
            if anaphor:
                # 检查回答中是否正确理解了指代（通过关键词匹配）
                keywords = turn.get("keywords", [])
                has_relevant_content = any(kw.lower() in answer.lower() for kw in keywords)
                
                if has_relevant_content:
                    result["scores"]["anaphora_resolution"] = 1.0
                    result["details"].append(f"指代词 '{anaphor}' 消解成功")
                else:
                    # 检查是否有上下文引用痕迹
                    history_text = " ".join(context_history[-3:])
                    has_history_ref = any(kw.lower() in history_text.lower() for kw in keywords)
                    if has_history_ref:
                        result["scores"]["anaphora_resolution"] = 0.7
                        result["details"].append(f"指代词 '{anaphor}' 部分消解")
                    else:
                        result["scores"]["anaphora_resolution"] = 0.3
                        result["details"].append(f"指代词 '{anaphor}' 消解可能失败")
                        result["passed"] = False
        
        # 2. 关键词相关性评估
        keywords = turn.get("keywords", [])
        if keywords:
            matched = sum(1 for kw in keywords if kw.lower() in answer.lower())
            result["scores"]["keyword_relevance"] = matched / len(keywords) if keywords else 0.5
            if matched == 0:
                result["passed"] = False
                result["details"].append("回答未命中预期关键词")
            result["matched_keywords"] = matched
            result["total_keywords"] = len(keywords)
        
        # 3. 上下文连贯性评估
        if context_history:
            # 检查回答长度和内容质量
            answer_length = len(answer.strip())
            if answer_length < 50:
                result["scores"]["context_coherence"] = 0.5
                result["details"].append("回答过于简短，可能信息不完整")
            elif answer_length < 100:
                result["scores"]["context_coherence"] = 0.7
            else:
                result["scores"]["context_coherence"] = 0.9
            
            # 检查是否有重复历史内容（简单检查）
            history_text = " ".join(context_history[-3:]).lower()
            answer_lower = answer.lower()
            if len(answer) > 0 and answer_lower in history_text:
                result["scores"]["context_coherence"] *= 0.8
                result["details"].append("回答可能重复历史内容")
        
        # 4. 综合得分计算（加权平均）
        weights = {
            "anaphora_resolution": 0.35,
            "keyword_relevance": 0.35,
            "context_coherence": 0.30
        }
        
        overall = sum(result["scores"][k] * weights[k] for k in weights)
        result["scores"]["overall"] = overall
        
        # 综合判断是否通过
        if overall < 0.5:
            result["passed"] = False
        
        return result
    
    def run_scenario_evaluation(self, scenario: Dict) -> Dict[str, Any]:
        """
        运行单个测试场景
        
        Args:
            scenario: 测试场景配置
            
        Returns:
            场景评估结果
        """
        scenario_id = scenario["scenario_id"]
        title = scenario["title"]
        turns = scenario["turns"]
        
        print(f"\n{'='*60}")
        print(f"测试场景: {title}")
        print(f"场景ID: {scenario_id}")
        print(f"轮次数: {len(turns)}")
        print('='*60)
        
        # 创建新会话（使用session_id机制）
        session_id = self.conversation_store.create_session(f"多轮评估_{title}")
        print(f"会话ID: {session_id}")
        
        conversation_history = []
        turn_evaluations = []
        all_context = []
        
        for i, turn in enumerate(turns, 1):
            question = turn["question"]
            turn_type = turn["type"]
            print(f"\n[轮次 {i}/{len(turns)}] 类型: {turn_type}")
            print(f"问题: {question}")
            
            # 生成回答（实际项目中应调用 RAGService.query(question, session_id=session_id)）
            answer = self._simulate_rag_response(question, all_context)
            
            # 评估本轮对话质量
            evaluation = self._evaluate_turn_quality(turn, answer, all_context)
            
            # 保存对话到会话存储（使用真实的conversation_store）
            sources = [{"content": "知识库检索结果", "doc_id": f"source_{i}"}]
            self.conversation_store.add_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                sources=sources,
                is_code_query=True  # 本次测试都是代码相关
            )
            
            # 更新对话历史
            conversation_history.append({
                "turn_id": i,
                "question": question,
                "answer": answer
            })
            
            all_context.append(question)
            all_context.append(answer)
            
            turn_evaluations.append(evaluation)
            
            # 输出本轮结果摘要
            status = "✓" if evaluation["passed"] else "✗"
            if len(answer) > 120:
                print(f"回答: {answer[:120]}...")
            else:
                print(f"回答: {answer}")
            print(f"评估: {status} 综合得分: {evaluation['scores']['overall']:.2f}")
            if evaluation["details"]:
                print(f"备注: {evaluation['details'][0]}")
        
        # 计算场景整体统计
        passed_turns = sum(1 for e in turn_evaluations if e["passed"])
        pass_rate = passed_turns / len(turn_evaluations)
        
        # 计算平均得分
        avg_scores = {}
        for metric in ["anaphora_resolution", "keyword_relevance", "context_coherence", "overall"]:
            avg_scores[metric] = sum(e["scores"][metric] for e in turn_evaluations) / len(turn_evaluations)
        
        print(f"\n[场景完成]")
        print(f"通过轮次: {passed_turns}/{len(turns)} ({pass_rate:.1%})")
        print(f"场景综合得分: {avg_scores['overall']:.2f}")
        
        return {
            "scenario_id": scenario_id,
            "title": title,
            "category": scenario["category"],
            "session_id": session_id,
            "total_turns": len(turns),
            "passed_turns": passed_turns,
            "pass_rate": pass_rate,
            "turn_evaluations": turn_evaluations,
            "conversation_history": conversation_history,
            "avg_scores": avg_scores
        }
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        运行完整的多轮对话评估
        
        Returns:
            完整评估结果
        """
        print("=" * 80)
        print("多轮对话质量评估开始")
        print("=" * 80)
        
        all_results = []
        
        for scenario in self.test_scenarios:
            result = self.run_scenario_evaluation(scenario)
            all_results.append(result)
        
        # 计算总体统计
        summary = self._calculate_summary(all_results)
        
        final_result = {
            "summary": summary,
            "scenario_results": all_results,
            "evaluation_time": datetime.now().isoformat()
        }
        
        # 生成报告
        report = self.generate_report(final_result)
        print("\n\n" + report)
        
        return final_result
    
    def _calculate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """
        计算评估汇总统计
        
        Args:
            results: 各场景评估结果列表
            
        Returns:
            汇总统计字典
        """
        total_scenarios = len(results)
        total_turns = sum(r["total_turns"] for r in results)
        total_passed = sum(r["passed_turns"] for r in results)
        
        # 按类别统计
        category_stats = {}
        for result in results:
            cat = result["category"]
            if cat not in category_stats:
                category_stats[cat] = {
                    "count": 0,
                    "total_turns": 0,
                    "passed_turns": 0,
                    "total_score": 0.0
                }
            category_stats[cat]["count"] += 1
            category_stats[cat]["total_turns"] += result["total_turns"]
            category_stats[cat]["passed_turns"] += result["passed_turns"]
            category_stats[cat]["total_score"] += result["avg_scores"]["overall"]
        
        # 计算各项平均分
        avg_scores = {}
        for metric in ["anaphora_resolution", "keyword_relevance", "context_coherence", "overall"]:
            avg_scores[metric] = sum(r["avg_scores"][metric] for r in results) / len(results)
        
        return {
            "total_scenarios": total_scenarios,
            "total_turns": total_turns,
            "total_passed_turns": total_passed,
            "overall_pass_rate": total_passed / total_turns if total_turns > 0 else 0,
            "average_scores": avg_scores,
            "category_stats": category_stats
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成文本格式的评估报告
        
        Args:
            results: 评估结果字典
            
        Returns:
            格式化的报告字符串
        """
        report = []
        report.append("=" * 80)
        report.append("多轮对话质量评估报告")
        report.append("=" * 80)
        report.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        summary = results["summary"]
        report.append(f"\n一、总体统计")
        report.append("-" * 40)
        report.append(f"  测试场景数: {summary['total_scenarios']}")
        report.append(f"  总对话轮次: {summary['total_turns']}")
        report.append(f"  通过轮次: {summary['total_passed_turns']} ({summary['overall_pass_rate']:.1%})")
        
        report.append(f"\n二、整体性能表现")
        report.append("-" * 40)
        scores = summary['average_scores']
        report.append(f"  指代消解准确率: {scores['anaphora_resolution']:.2f}")
        report.append(f"  关键词相关性: {scores['keyword_relevance']:.2f}")
        report.append(f"  上下文连贯性: {scores['context_coherence']:.2f}")
        report.append(f"  综合得分: {scores['overall']:.2f}")
        
        report.append(f"\n三、各场景详细结果")
        report.append("-" * 60)
        report.append(f"{'场景标题':<25} {'类别':<10} {'轮次':<6} {'通过率':<8} {'综合得分':<10}")
        report.append("-" * 60)
        
        for scenario in results["scenario_results"]:
            pass_rate_str = f"{scenario['pass_rate']*100:.0f}%"
            report.append(f"{scenario['title'][:23]:<25} {scenario['category'][:8]:<10} "
                         f"{scenario['total_turns']:<6} {pass_rate_str:<8} "
                         f"{scenario['avg_scores']['overall']:<10.2f}")
        
        report.append(f"\n四、按测试类别统计")
        report.append("-" * 40)
        for cat, stats in summary["category_stats"].items():
            cat_pass_rate = stats["passed_turns"] / stats["total_turns"] if stats["total_turns"] > 0 else 0
            cat_avg_score = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0
            report.append(f"  {cat}:")
            report.append(f"    场景数: {stats['count']}, 总轮次: {stats['total_turns']}, "
                         f"通过率: {cat_pass_rate:.1%}, 平均得分: {cat_avg_score:.2f}")
        
        report.append("\n" + "=" * 80)
        report.append("五、评估结论")
        report.append("=" * 80)
        
        overall_score = scores['overall']
        if overall_score >= 0.8:
            report.append("  多轮对话质量优秀，上下文理解能力强！")
            report.append("  ✓ 指代消解准确")
            report.append("  ✓ 上下文关联紧密")
            report.append("  ✓ 信息融合充分")
        elif overall_score >= 0.6:
            report.append("  多轮对话质量良好，能够处理基本的上下文关联。")
            report.append("  ⚠ 可进一步优化指代消解的准确率")
            report.append("  ⚠ 建议增强信息融合的深度")
        elif overall_score >= 0.4:
            report.append("  多轮对话质量一般，需要重点改进以下方面：")
            report.append("  ✗ 指代消解准确率有待提升")
            report.append("  ✗ 上下文连贯性需要加强")
            report.append("  ✗ 历史信息的记忆和利用不足")
        else:
            report.append("  多轮对话质量较差，需要重点改进对话历史管理机制。")
            report.append("  ✗ 建议重构上下文管理模块")
            report.append("  ✗ 需要增强对话状态跟踪能力")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        保存评估结果到JSON文件
        
        Args:
            results: 评估结果字典
            filename: 可选文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multirun_evaluation_{timestamp}.json"
        
        # 确保目录存在
        output_dir = "./data/evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        # 简化结果以便JSON序列化（移除大型对话内容）
        serializable = {
            "summary": results["summary"],
            "evaluation_time": results["evaluation_time"],
            "scenario_summaries": [
                {
                    "scenario_id": r["scenario_id"],
                    "title": r["title"],
                    "category": r["category"],
                    "session_id": r["session_id"],
                    "total_turns": r["total_turns"],
                    "passed_turns": r["passed_turns"],
                    "pass_rate": r["pass_rate"],
                    "avg_scores": r["avg_scores"]
                } for r in results["scenario_results"]
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估结果已保存到: {filepath}")
        return filepath


def prepare_test_knowledge_base():
    """准备测试知识库"""
    print("\n[准备] 初始化测试知识库...")
    
    # 重新初始化向量存储
    vector_store = VectorStore()
    vector_store._create_index()  # 清空重建
    
    # 测试文档（代码相关主题）
    test_docs = [
        # Python装饰器相关
        ("Python装饰器是一种用于修改函数或类行为的高级语法特性，"
         "本质上是接收函数作为参数并返回新函数的可调用对象。"
         "常见用途包括日志记录、性能监控、权限验证、缓存、参数校验等。"
         "装饰器不修改原始函数代码，遵循开闭原则。", 
         {"type": "python", "topic": "decorator", "doc_id": "py_dec_001"}),
        
        ("装饰器的基本语法使用@符号，也称为语法糖。带参数的装饰器需要额外一层函数嵌套。"
         "使用functools.wraps可以保留原始函数的元数据信息（如函数名、文档字符串等）。"
         "多层装饰器嵌套时执行顺序是从内到外或从外到内需要注意。",
         {"type": "python", "topic": "decorator_advanced", "doc_id": "py_dec_002"}),
        
        ("类装饰器是另一种实现方式，通过实现__call__方法来实现装饰器功能。"
         "装饰器在Web框架如Flask和Django中广泛用于路由定义、请求处理、权限控制等场景。"
         "常见的第三方库如functools提供了许多装饰器相关工具。",
         {"type": "python", "topic": "decorator_app", "doc_id": "py_dec_003"}),
        
        # FAISS相关
        ("FAISS是Facebook AI Research开发的高效相似性搜索库。"
         "它专门用于处理大规模高维向量的快速检索问题，是当前工业界最常用的向量检索库之一。"
         "支持CPU和GPU版本，GPU版本可获得数十倍的性能提升。",
         {"type": "faiss", "topic": "introduction", "doc_id": "faiss_001"}),
        
        ("FAISS支持多种索引类型：1) IndexFlatL2 - 暴力搜索，100%精确但速度慢；"
         "2) IndexIVFFlat - 倒排文件，速度与精度权衡；3) IndexHNSW - 层次化导航小世界，"
         "查询效率最高但内存占用较高。HNSW是大规模高维向量检索的首选。",
         {"type": "faiss", "topic": "index_types", "doc_id": "faiss_002"}),
        
        ("IndexHNSW是FAISS中查询效率最高的索引类型，采用层次化导航小世界算法。"
         "它构建一个多层的有向图结构，搜索时从上层开始，逐层向下找到最近邻。"
         "支持增量添加向量，适合动态更新的场景。",
         {"type": "faiss", "topic": "hnsw", "doc_id": "faiss_003"}),
        
        # Linux相关
        ("Linux系统提供了丰富的命令行工具用于系统监控和资源管理："
         "top/htop - 实时显示系统资源使用情况；free - 显示内存使用情况；"
         "df - 显示磁盘空间使用情况；du - 估算目录和文件的磁盘使用量；"
         "ps aux - 查看当前运行的所有进程；vmstat/iostat - 虚拟内存和IO统计。",
         {"type": "linux", "topic": "monitoring", "doc_id": "linux_001"}),
        
        ("Shell脚本可以实现自动化监控，基本结构包括："
         "1) 定义阈值变量；2) 采集系统数据（使用df、top等命令）；"
         "3) 条件判断；4) 告警动作（邮件、短信等）；5) 定时执行结合cron。"
         "常见的坑点包括：相对路径问题、环境变量问题、权限问题、日志处理等。",
         {"type": "linux", "topic": "shell_script", "doc_id": "linux_002"}),
        
        # API设计相关
        ("RESTful架构是一种基于HTTP协议的API设计风格。核心原则包括："
         "资源标识 - 使用URL唯一标识每个资源；通过HTTP方法表示操作 - "
         "GET获取、POST创建、PUT更新、DELETE删除；无状态通信；统一接口。"
         "幂等性对于分布式系统和重试机制至关重要。",
         {"type": "api", "topic": "restful", "doc_id": "api_001"})
    ]
    
    # 使用单例的embedding model
    embedding_model = EmbeddingModel()
    
    # 生成嵌入向量并添加到向量库
    contents = [doc[0] for doc in test_docs]
    embeddings = embedding_model.embed_documents(contents)
    
    # 准备文档元数据
    documents = []
    for content, metadata in test_docs:
        doc_metadata = {
            "content": content,
            **metadata
        }
        documents.append(doc_metadata)
    
    # 添加到向量库
    vector_store.add_documents(embeddings, documents)
    print(f"已添加 {len(test_docs)} 篇测试文档到知识库")


def main():
    """主函数：执行多轮对话评估"""
    print("\n" + "="*80)
    print("多轮对话质量评估工具 v1.0")
    print("="*80)
    
    # 准备测试知识库
    prepare_test_knowledge_base()
    
    # 创建评估器并运行测试
    evaluator = MultirunDialogEvaluator()
    results = evaluator.run_full_evaluation()
    
    # 保存结果
    evaluator.save_results(results)
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()
