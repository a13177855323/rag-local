#!/usr/bin/env python3
"""
RAG系统检索质量评估脚本

功能：
1. 构造20条不同难度级别的测试查询（精确匹配、模糊匹配、语义理解）
2. 计算评估指标：Hit Rate@k、MRR、NDCG
3. 批量测试：支持不同top_k参数下的性能测试
4. 结果可视化：生成文本格式性能报告

约束：
- 复用VectorStore和EmbeddingModel组件
- CPU环境下高效运行
- 兼容现有FAISS索引结构

Author: RAG Testing Team
Date: 2026-03-24
"""

import os
import sys
import json
import time
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.services.vector_store import VectorStore
from backend.models.embedding_model import EmbeddingModel


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class TestQuery:
    """测试查询数据类
    
    Attributes:
        query_id: 查询唯一标识
        query_text: 查询文本
        relevant_doc_ids: 相关文档ID集合
        difficulty: 难度级别 (exact_match/fuzzy_match/semantic_understanding)
        category: 查询类别 (technical/conceptual/how_to/comparison)
        description: 测试用例描述
    """
    query_id: str
    query_text: str
    relevant_doc_ids: Set[str] = field(default_factory=set)
    difficulty: str = "exact_match"  # exact_match, fuzzy_match, semantic_understanding
    category: str = "technical"  # technical, conceptual, how_to, comparison
    description: str = ""


@dataclass
class RetrievalResult:
    """检索结果数据类
    
    Attributes:
        query_id: 查询ID
        query_text: 查询文本
        retrieved_docs: 检索到的文档列表 [(doc_id, score, rank), ...]
        relevant_docs: 相关文档ID集合
        metrics: 该查询的各项指标
        latency_ms: 检索延迟（毫秒）
    """
    query_id: str
    query_text: str
    retrieved_docs: List[Tuple[str, float, int]] = field(default_factory=list)
    relevant_docs: Set[str] = field(default_factory=set)
    metrics: Dict = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class EvaluationMetrics:
    """评估指标数据类
    
    Attributes:
        hit_rate_at_k: Hit Rate@k 指标
        mrr: Mean Reciprocal Rank
        ndcg: Normalized Discounted Cumulative Gain
        avg_latency_ms: 平均检索延迟
    """
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: float = 0.0
    avg_latency_ms: float = 0.0


# ============================================================================
# 测试数据集构造器
# ============================================================================

class TestDatasetBuilder:
    """测试数据集构造器
    
    生成20条不同难度级别的测试查询，涵盖：
    - 精确匹配（Exact Match）：查询词直接出现在文档中
    - 模糊匹配（Fuzzy Match）：查询词与文档相关但不完全相同
    - 语义理解（Semantic Understanding）：需要理解语义才能匹配
    """
    
    def __init__(self):
        self.test_queries: List[TestQuery] = []
        self._build_dataset()
    
    def _build_dataset(self):
        """构建测试数据集"""
        
        # ========== 精确匹配测试（7条）==========
        exact_match_queries = [
            TestQuery(
                query_id="exact_001",
                query_text="什么是FAISS索引",
                relevant_doc_ids={"doc_faiss_intro", "doc_vector_db"},
                difficulty="exact_match",
                category="technical",
                description="技术术语精确匹配"
            ),
            TestQuery(
                query_id="exact_002",
                query_text="Python中的列表推导式",
                relevant_doc_ids={"doc_python_list", "doc_python_advanced"},
                difficulty="exact_match",
                category="technical",
                description="编程语言特性查询"
            ),
            TestQuery(
                query_id="exact_003",
                query_text="Transformer架构原理",
                relevant_doc_ids={"doc_transformer", "doc_attention_mechanism"},
                difficulty="exact_match",
                category="technical",
                description="深度学习架构查询"
            ),
            TestQuery(
                query_id="exact_004",
                query_text="REST API设计规范",
                relevant_doc_ids={"doc_rest_api", "doc_api_design"},
                difficulty="exact_match",
                category="technical",
                description="API设计规范查询"
            ),
            TestQuery(
                query_id="exact_005",
                query_text="Docker容器化部署",
                relevant_doc_ids={"doc_docker", "doc_deployment"},
                difficulty="exact_match",
                category="technical",
                description="DevOps技术查询"
            ),
            TestQuery(
                query_id="exact_006",
                query_text="SQL性能优化技巧",
                relevant_doc_ids={"doc_sql_optimization", "db_performance"},
                difficulty="exact_match",
                category="technical",
                description="数据库优化查询"
            ),
            TestQuery(
                query_id="exact_007",
                query_text="Git分支管理策略",
                relevant_doc_ids={"doc_git_workflow", "doc_version_control"},
                difficulty="exact_match",
                category="technical",
                description="版本控制查询"
            ),
        ]
        
        # ========== 模糊匹配测试（7条）==========
        fuzzy_match_queries = [
            TestQuery(
                query_id="fuzzy_001",
                query_text="如何提高代码运行速度",
                relevant_doc_ids={"doc_code_optimization", "doc_performance_tuning", "doc_profiling"},
                difficulty="fuzzy_match",
                category="how_to",
                description="性能优化相关（模糊表达）"
            ),
            TestQuery(
                query_id="fuzzy_002",
                query_text="神经网络训练不收敛怎么办",
                relevant_doc_ids={"doc_training_tips", "doc_gradient_problems", "doc_optimization"},
                difficulty="fuzzy_match",
                category="how_to",
                description="训练问题排查（问题描述）"
            ),
            TestQuery(
                query_id="fuzzy_003",
                query_text="数据存储方案对比",
                relevant_doc_ids={"doc_database_comparison", "doc_storage_solutions", "doc_nosql"},
                difficulty="fuzzy_match",
                category="comparison",
                description="技术选型对比"
            ),
            TestQuery(
                query_id="fuzzy_004",
                query_text="前后端分离的优势",
                relevant_doc_ids={"doc_frontend_backend", "doc_architecture", "doc_microservices"},
                difficulty="fuzzy_match",
                category="conceptual",
                description="架构概念理解"
            ),
            TestQuery(
                query_id="fuzzy_005",
                query_text="模型部署到生产环境",
                relevant_doc_ids={"doc_model_serving", "doc_mlops", "doc_deployment"},
                difficulty="fuzzy_match",
                category="how_to",
                description="ML工程化部署"
            ),
            TestQuery(
                query_id="fuzzy_006",
                query_text="处理大规模数据的方法",
                relevant_doc_ids={"doc_big_data", "doc_distributed_computing", "doc_spark"},
                difficulty="fuzzy_match",
                category="how_to",
                description="大数据处理"
            ),
            TestQuery(
                query_id="fuzzy_007",
                query_text="代码质量保证措施",
                relevant_doc_ids={"doc_code_review", "doc_testing", "doc_ci_cd"},
                difficulty="fuzzy_match",
                category="conceptual",
                description="软件工程实践"
            ),
        ]
        
        # ========== 语义理解测试（6条）==========
        semantic_queries = [
            TestQuery(
                query_id="semantic_001",
                query_text="让机器理解人类语言的技术",
                relevant_doc_ids={"doc_nlp", "doc_transformer", "doc_bert"},
                difficulty="semantic_understanding",
                category="conceptual",
                description="NLP概念理解（非关键词匹配）"
            ),
            TestQuery(
                query_id="semantic_002",
                query_text="不用写代码就能创建应用的平台",
                relevant_doc_ids={"doc_low_code", "doc_no_code", "doc_app_builder"},
                difficulty="semantic_understanding",
                category="conceptual",
                description="低代码概念理解"
            ),
            TestQuery(
                query_id="semantic_003",
                query_text="根据用户喜好推荐商品的算法",
                relevant_doc_ids={"doc_recommendation", "doc_collaborative_filtering", "doc_matrix_factorization"},
                difficulty="semantic_understanding",
                category="conceptual",
                description="推荐系统概念"
            ),
            TestQuery(
                query_id="semantic_004",
                query_text="自动发现数据中的模式",
                relevant_doc_ids={"doc_machine_learning", "doc_pattern_recognition", "doc_clustering"},
                difficulty="semantic_understanding",
                category="conceptual",
                description="机器学习概念"
            ),
            TestQuery(
                query_id="semantic_005",
                query_text="保护数据不被未授权访问",
                relevant_doc_ids={"doc_security", "doc_encryption", "doc_authentication"},
                difficulty="semantic_understanding",
                category="conceptual",
                description="安全概念理解"
            ),
            TestQuery(
                query_id="semantic_006",
                query_text="多台计算机协同工作",
                relevant_doc_ids={"doc_distributed_systems", "doc_cluster_computing", "doc_load_balancing"},
                difficulty="semantic_understanding",
                category="conceptual",
                description="分布式系统概念"
            ),
        ]
        
        # 合并所有查询
        self.test_queries = (
            exact_match_queries + 
            fuzzy_match_queries + 
            semantic_queries
        )
    
    def get_queries_by_difficulty(self, difficulty: str) -> List[TestQuery]:
        """按难度获取测试查询"""
        return [q for q in self.test_queries if q.difficulty == difficulty]
    
    def get_queries_by_category(self, category: str) -> List[TestQuery]:
        """按类别获取测试查询"""
        return [q for q in self.test_queries if q.category == category]
    
    def get_all_queries(self) -> List[TestQuery]:
        """获取所有测试查询"""
        return self.test_queries
    
    def export_to_json(self, filepath: str):
        """导出测试集到JSON文件"""
        data = []
        for q in self.test_queries:
            q_dict = asdict(q)
            q_dict['relevant_doc_ids'] = list(q_dict['relevant_doc_ids'])
            data.append(q_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"测试集已导出到: {filepath}")


# ============================================================================
# 评估指标计算器
# ============================================================================

class MetricsCalculator:
    """检索质量评估指标计算器
    
    支持指标：
    - Hit Rate@k: 前k个结果中至少有一个相关文档的比例
    - MRR (Mean Reciprocal Rank): 平均倒数排名
    - NDCG (Normalized Discounted Cumulative Gain): 归一化折损累计增益
    """
    
    @staticmethod
    def calculate_hit_rate_at_k(
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        k: int
    ) -> float:
        """计算 Hit Rate@k
        
        Hit Rate@k = 1 如果前k个结果中有相关文档，否则为 0
        
        Args:
            retrieved_doc_ids: 检索到的文档ID列表（按相关性排序）
            relevant_doc_ids: 相关文档ID集合
            k: 考虑的前k个结果
            
        Returns:
            1.0 表示命中，0.0 表示未命中
        """
        if not relevant_doc_ids:
            return 0.0
        
        # 取前k个结果
        top_k_docs = set(retrieved_doc_ids[:k])
        
        # 检查是否有交集
        if top_k_docs & relevant_doc_ids:
            return 1.0
        return 0.0
    
    @staticmethod
    def calculate_mrr(
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str]
    ) -> float:
        """计算 MRR (Mean Reciprocal Rank)
        
        MRR = 1 / rank_of_first_relevant_doc
        如果没有相关文档，MRR = 0
        
        Args:
            retrieved_doc_ids: 检索到的文档ID列表（按相关性排序）
            relevant_doc_ids: 相关文档ID集合
            
        Returns:
            倒数排名值 (0.0 ~ 1.0]
        """
        if not relevant_doc_ids:
            return 0.0
        
        # 找到第一个相关文档的排名
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def calculate_dcg(relevances: List[float]) -> float:
        """计算 DCG (Discounted Cumulative Gain)
        
        DCG = sum(rel_i / log2(i + 2))
        
        Args:
            relevances: 相关性分数列表
            
        Returns:
            DCG值
        """
        dcg = 0.0
        for i, rel in enumerate(relevances):
            # i从0开始，所以位置是i+1，分母是log2(i+2)
            dcg += rel / math.log2(i + 2)
        return dcg
    
    @staticmethod
    def calculate_ndcg(
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        k: int = 10
    ) -> float:
        """计算 NDCG (Normalized Discounted Cumulative Gain)
        
        NDCG = DCG / IDCG
        其中IDCG是理想情况下的DCG（所有相关文档都在最前面）
        
        Args:
            retrieved_doc_ids: 检索到的文档ID列表
            relevant_doc_ids: 相关文档ID集合
            k: 考虑的前k个结果
            
        Returns:
            NDCG值 (0.0 ~ 1.0]
        """
        if not relevant_doc_ids:
            return 0.0
        
        # 构建相关性列表（二元相关性：1表示相关，0表示不相关）
        relevances = []
        for doc_id in retrieved_doc_ids[:k]:
            relevances.append(1.0 if doc_id in relevant_doc_ids else 0.0)
        
        # 计算DCG
        dcg = MetricsCalculator.calculate_dcg(relevances)
        
        # 计算IDCG（理想情况：所有相关文档都在最前面）
        num_relevant = min(len(relevant_doc_ids), k)
        ideal_relevances = [1.0] * num_relevant + [0.0] * (k - num_relevant)
        idcg = MetricsCalculator.calculate_dcg(ideal_relevances)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def calculate_all_metrics(
        retrieved_doc_ids: List[str],
        relevant_doc_ids: Set[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """计算所有评估指标
        
        Args:
            retrieved_doc_ids: 检索到的文档ID列表
            relevant_doc_ids: 相关文档ID集合
            k_values: 要计算的k值列表
            
        Returns:
            包含所有指标的字典
        """
        metrics = {
            'hit_rate_at_k': {},
            'mrr': MetricsCalculator.calculate_mrr(retrieved_doc_ids, relevant_doc_ids),
            'ndcg_at_10': MetricsCalculator.calculate_ndcg(retrieved_doc_ids, relevant_doc_ids, k=10)
        }
        
        # 计算不同k值的Hit Rate
        for k in k_values:
            metrics['hit_rate_at_k'][f'@{k}'] = MetricsCalculator.calculate_hit_rate_at_k(
                retrieved_doc_ids, relevant_doc_ids, k
            )
        
        return metrics


# ============================================================================
# 检索质量评估器
# ============================================================================

class RetrievalQualityEvaluator:
    """检索质量评估器主类
    
    功能：
    1. 批量执行检索测试
    2. 计算各项评估指标
    3. 生成性能报告
    """
    
    def __init__(self):
        """初始化评估器"""
        self.vector_store = VectorStore()
        self.embedding_model = EmbeddingModel()
        self.dataset_builder = TestDatasetBuilder()
        self.results: List[RetrievalResult] = []
        
    def evaluate_single_query(
        self,
        query: TestQuery,
        top_k: int = 10
    ) -> RetrievalResult:
        """评估单个查询
        
        Args:
            query: 测试查询
            top_k: 检索结果数量
            
        Returns:
            检索结果对象
        """
        # 记录开始时间
        start_time = time.time()
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_text(query.query_text)
        
        # 执行检索
        search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # 计算延迟
        latency_ms = (time.time() - start_time) * 1000
        
        # 构建检索到的文档列表
        retrieved_docs = []
        for rank, (doc_metadata, score) in enumerate(search_results, start=1):
            doc_id = doc_metadata.get('id', doc_metadata.get('metadata', {}).get('id', f'unknown_{rank}'))
            retrieved_docs.append((doc_id, score, rank))
        
        # 提取文档ID列表
        retrieved_doc_ids = [doc[0] for doc in retrieved_docs]
        
        # 计算指标
        metrics = MetricsCalculator.calculate_all_metrics(
            retrieved_doc_ids,
            query.relevant_doc_ids,
            k_values=[1, 3, 5, 10]
        )
        
        # 构建结果对象
        result = RetrievalResult(
            query_id=query.query_id,
            query_text=query.query_text,
            retrieved_docs=retrieved_docs,
            relevant_docs=query.relevant_doc_ids,
            metrics=metrics,
            latency_ms=latency_ms
        )
        
        return result
    
    def run_batch_evaluation(
        self,
        queries: Optional[List[TestQuery]] = None,
        top_k_values: List[int] = [5, 10]
    ) -> Dict:
        """批量评估测试
        
        Args:
            queries: 测试查询列表，默认使用全部
            top_k_values: 要测试的不同top_k值
            
        Returns:
            评估结果字典
        """
        if queries is None:
            queries = self.dataset_builder.get_all_queries()
        
        print(f"\n开始批量评估: {len(queries)} 条查询, top_k 值: {top_k_values}")
        print("=" * 70)
        
        all_results = {}
        
        for top_k in top_k_values:
            print(f"\n测试 top_k={top_k}...")
            
            results = []
            for i, query in enumerate(queries, 1):
                result = self.evaluate_single_query(query, top_k=top_k)
                results.append(result)
                
                # 显示进度
                if i % 5 == 0:
                    print(f"  进度: {i}/{len(queries)}")
            
            all_results[f'top_k_{top_k}'] = results
        
        return all_results
    
    def calculate_aggregate_metrics(
        self,
        results: List[RetrievalResult]
    ) -> Dict:
        """计算聚合指标
        
        Args:
            results: 检索结果列表
            
        Returns:
            聚合指标字典
        """
        if not results:
            return {}
        
        # 按难度分类统计
        difficulty_groups = defaultdict(list)
        for r in results:
            # 从query_id推断难度
            if 'exact' in r.query_id:
                difficulty_groups['exact_match'].append(r)
            elif 'fuzzy' in r.query_id:
                difficulty_groups['fuzzy_match'].append(r)
            elif 'semantic' in r.query_id:
                difficulty_groups['semantic_understanding'].append(r)
        
        # 计算整体指标
        aggregate = {
            'overall': self._compute_metrics_stats(results),
            'by_difficulty': {}
        }
        
        # 按难度计算指标
        for difficulty, group_results in difficulty_groups.items():
            aggregate['by_difficulty'][difficulty] = self._compute_metrics_stats(group_results)
        
        return aggregate
    
    def _compute_metrics_stats(self, results: List[RetrievalResult]) -> Dict:
        """计算指标统计值"""
        if not results:
            return {}
        
        # 收集所有指标
        hit_rates = {k: [] for k in ['@1', '@3', '@5', '@10']}
        mrrs = []
        ndcgs = []
        latencies = []
        
        for r in results:
            for k, v in r.metrics['hit_rate_at_k'].items():
                hit_rates[k].append(v)
            mrrs.append(r.metrics['mrr'])
            ndcgs.append(r.metrics['ndcg_at_10'])
            latencies.append(r.latency_ms)
        
        return {
            'hit_rate_at_k': {
                k: round(np.mean(v), 4) for k, v in hit_rates.items()
            },
            'mrr': round(np.mean(mrrs), 4),
            'ndcg_at_10': round(np.mean(ndcgs), 4),
            'avg_latency_ms': round(np.mean(latencies), 2),
            'total_queries': len(results)
        }


# ============================================================================
# 报告生成器
# ============================================================================

class ReportGenerator:
    """测试报告生成器
    
    生成文本格式的性能对比报告
    """
    
    @staticmethod
    def generate_text_report(
        evaluation_results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """生成文本格式报告
        
        Args:
            evaluation_results: 评估结果字典
            output_path: 输出文件路径
            
        Returns:
            报告文本内容
        """
        lines = []
        
        # 报告标题
        lines.append("=" * 80)
        lines.append(" " * 25 + "RAG系统检索质量评估报告")
        lines.append("=" * 80)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 测试配置
        lines.append("-" * 80)
        lines.append("测试配置")
        lines.append("-" * 80)
        lines.append(f"测试查询总数: 20 条")
        lines.append("  - 精确匹配: 7 条")
        lines.append("  - 模糊匹配: 7 条")
        lines.append("  - 语义理解: 6 条")
        lines.append("")
        
        # 各top_k配置下的结果
        for config_name, results in evaluation_results.items():
            top_k = config_name.replace('top_k_', '')
            
            lines.append("-" * 80)
            lines.append(f"Top-K = {top_k} 性能评估")
            lines.append("-" * 80)
            
            # 计算聚合指标
            evaluator = RetrievalQualityEvaluator()
            aggregate = evaluator.calculate_aggregate_metrics(results)
            
            # 整体指标
            overall = aggregate.get('overall', {})
            lines.append("\n【整体性能指标】")
            lines.append(f"  Hit Rate@1:  {overall.get('hit_rate_at_k', {}).get('@1', 0):.2%}")
            lines.append(f"  Hit Rate@3:  {overall.get('hit_rate_at_k', {}).get('@3', 0):.2%}")
            lines.append(f"  Hit Rate@5:  {overall.get('hit_rate_at_k', {}).get('@5', 0):.2%}")
            lines.append(f"  Hit Rate@10: {overall.get('hit_rate_at_k', {}).get('@10', 0):.2%}")
            lines.append(f"  MRR:         {overall.get('mrr', 0):.4f}")
            lines.append(f"  NDCG@10:     {overall.get('ndcg_at_10', 0):.4f}")
            lines.append(f"  平均延迟:    {overall.get('avg_latency_ms', 0):.2f} ms")
            
            # 按难度分类
            lines.append("\n【按难度分类】")
            by_difficulty = aggregate.get('by_difficulty', {})
            
            for difficulty in ['exact_match', 'fuzzy_match', 'semantic_understanding']:
                if difficulty in by_difficulty:
                    stats = by_difficulty[difficulty]
                    diff_name = {
                        'exact_match': '精确匹配',
                        'fuzzy_match': '模糊匹配',
                        'semantic_understanding': '语义理解'
                    }.get(difficulty, difficulty)
                    
                    lines.append(f"\n  {diff_name}:")
                    lines.append(f"    Hit Rate@5:  {stats.get('hit_rate_at_k', {}).get('@5', 0):.2%}")
                    lines.append(f"    MRR:         {stats.get('mrr', 0):.4f}")
                    lines.append(f"    NDCG@10:     {stats.get('ndcg_at_10', 0):.4f}")
            
            lines.append("")
        
        # 性能对比
        if len(evaluation_results) > 1:
            lines.append("-" * 80)
            lines.append("Top-K 参数性能对比")
            lines.append("-" * 80)
            lines.append("")
            
            # 表头
            header_cols = [f"{k.replace('top_k_', 'K='):<12}" for k in evaluation_results.keys()]
            lines.append(f"{'指标':<20} {' | '.join(header_cols)}")
            lines.append("-" * 80)
            
            # 指标行
            metrics_to_compare = ['hit_rate_at_k', 'mrr', 'ndcg_at_10', 'avg_latency_ms']
            metric_names = {
                'hit_rate_at_k': 'Hit Rate@5',
                'mrr': 'MRR',
                'ndcg_at_10': 'NDCG@10',
                'avg_latency_ms': '平均延迟(ms)'
            }
            
            for metric in metrics_to_compare:
                row = [f"{metric_names.get(metric, metric):<20}"]
                for config_name, results in evaluation_results.items():
                    evaluator = RetrievalQualityEvaluator()
                    aggregate = evaluator.calculate_aggregate_metrics(results)
                    overall = aggregate.get('overall', {})
                    
                    if metric == 'hit_rate_at_k':
                        value = overall.get('hit_rate_at_k', {}).get('@5', 0)
                        row.append(f"{value:.2%}".ljust(12))
                    elif metric == 'avg_latency_ms':
                        value = overall.get(metric, 0)
                        row.append(f"{value:.2f}".ljust(12))
                    else:
                        value = overall.get(metric, 0)
                        row.append(f"{value:.4f}".ljust(12))
                
                lines.append(" | ".join(row))
            
            lines.append("")
        
        # 总结与建议
        lines.append("-" * 80)
        lines.append("总结与建议")
        lines.append("-" * 80)
        lines.append("")
        lines.append("1. 评估指标说明:")
        lines.append("   - Hit Rate@k: 前k个结果中命中的比例，越高越好")
        lines.append("   - MRR: 平均倒数排名，范围(0,1]，越高越好")
        lines.append("   - NDCG@10: 归一化折损累计增益，范围[0,1]，越高越好")
        lines.append("")
        lines.append("2. 性能基准参考:")
        lines.append("   - 优秀: Hit Rate@5 > 80%, MRR > 0.6, NDCG@10 > 0.7")
        lines.append("   - 良好: Hit Rate@5 > 60%, MRR > 0.4, NDCG@10 > 0.5")
        lines.append("   - 需改进: Hit Rate@5 < 60%, MRR < 0.4, NDCG@10 < 0.5")
        lines.append("")
        lines.append("3. 优化建议:")
        lines.append("   - 如精确匹配性能低，检查索引构建和嵌入质量")
        lines.append("   - 如语义理解性能低，考虑使用更强的嵌入模型")
        lines.append("   - 如延迟过高，考虑使用更高效的索引结构（如HNSW）")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("报告生成完成")
        lines.append("=" * 80)
        
        report_text = "\n".join(lines)
        
        # 保存到文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n报告已保存到: {output_path}")
        
        return report_text
    
    @staticmethod
    def generate_json_report(
        evaluation_results: Dict,
        output_path: str
    ):
        """生成JSON格式详细报告
        
        Args:
            evaluation_results: 评估结果字典
            output_path: 输出文件路径
        """
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'configurations': {}
        }
        
        for config_name, results in evaluation_results.items():
            evaluator = RetrievalQualityEvaluator()
            aggregate = evaluator.calculate_aggregate_metrics(results)
            
            # 转换结果为可序列化格式
            detailed_results = []
            for r in results:
                detailed_results.append({
                    'query_id': r.query_id,
                    'query_text': r.query_text,
                    'retrieved_doc_ids': [d[0] for d in r.retrieved_docs],
                    'relevant_doc_ids': list(r.relevant_docs),
                    'metrics': r.metrics,
                    'latency_ms': r.latency_ms
                })
            
            report_data['configurations'][config_name] = {
                'aggregate_metrics': aggregate,
                'detailed_results': detailed_results
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"JSON报告已保存到: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 执行完整的检索质量评估"""
    
    print("=" * 80)
    print(" " * 25 + "RAG系统检索质量评估")
    print("=" * 80)
    
    # 初始化评估器
    evaluator = RetrievalQualityEvaluator()
    
    # 检查向量数据库状态
    doc_count = evaluator.vector_store.get_document_count()
    print(f"\n向量数据库状态:")
    print(f"  - 文档数量: {doc_count}")
    
    if doc_count == 0:
        print("\n警告: 向量数据库为空，请先导入测试文档!")
        print("提示: 使用 test_import_documents() 函数导入示例文档")
        return
    
    # 执行批量评估
    print("\n开始检索质量评估...")
    evaluation_results = evaluator.run_batch_evaluation(
        queries=None,  # 使用全部20条测试查询
        top_k_values=[5, 10]  # 测试不同top_k值
    )
    
    # 生成报告
    print("\n生成评估报告...")
    
    # 文本报告
    report_gen = ReportGenerator()
    report_text = report_gen.generate_text_report(
        evaluation_results,
        output_path='backend/tests/retrieval_evaluation_report.txt'
    )
    
    # 打印报告
    print("\n" + report_text)
    
    # JSON详细报告
    report_gen.generate_json_report(
        evaluation_results,
        output_path='backend/tests/retrieval_evaluation_report.json'
    )
    
    print("\n评估完成!")


def test_import_documents():
    """导入测试文档到向量数据库
    
    用于测试前的数据准备
    """
    print("=" * 80)
    print("导入测试文档")
    print("=" * 80)
    
    from backend.models.embedding_model import EmbeddingModel
    from backend.services.vector_store import VectorStore
    
    # 先初始化 embedding_model 以获取正确的维度
    embedding_model = EmbeddingModel()
    
    # 然后初始化 vector_store（会自动使用正确的维度）
    vector_store = VectorStore()
    
    # 测试文档集
    test_documents = [
        {"id": "doc_faiss_intro", "content": "FAISS是Facebook开发的向量相似度搜索库，支持高效的最近邻搜索。FAISS索引可以存储高维向量并快速检索相似向量。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_vector_db", "content": "向量数据库用于存储和检索高维向量数据。FAISS是一种流行的向量数据库实现，支持多种索引类型。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_python_list", "content": "Python列表推导式是一种简洁的创建列表的方式。例如：[x for x in range(10)] 创建一个0到9的列表。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_python_advanced", "content": "Python高级特性包括生成器、装饰器、上下文管理器等。列表推导式是Pythonic代码的重要组成部分。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_transformer", "content": "Transformer架构是深度学习的里程碑，使用自注意力机制处理序列数据。BERT和GPT都基于Transformer。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_attention_mechanism", "content": "注意力机制允许模型关注输入的不同部分。自注意力是Transformer的核心，计算序列中各位置的相关性。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_rest_api", "content": "REST API设计规范包括使用HTTP方法表示操作、URL表示资源、状态码表示结果等原则。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_api_design", "content": "良好的API设计应该简洁、一致、可预测。RESTful API是目前最流行的Web API设计风格。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_docker", "content": "Docker容器化部署将应用及其依赖打包成容器镜像，确保环境一致性。Docker是DevOps的重要工具。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_deployment", "content": "应用部署涉及将代码发布到生产环境。容器化部署使用Docker等技术简化部署流程。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_sql_optimization", "content": "SQL性能优化技巧包括创建索引、优化查询语句、避免全表扫描等。良好的索引设计是性能的关键。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "db_performance", "content": "数据库性能优化涉及查询优化、索引设计、连接池配置等方面。慢查询日志是诊断性能问题的重要工具。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_git_workflow", "content": "Git分支管理策略包括Git Flow、GitHub Flow等。合理的分支策略支持团队协作和持续集成。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_version_control", "content": "版本控制是软件开发的基础实践。Git是目前最流行的分布式版本控制系统。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_code_optimization", "content": "代码优化包括算法优化、数据结构选择、并行计算等。性能分析工具帮助识别瓶颈。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_performance_tuning", "content": "性能调优是系统优化的重要环节。通过分析和测试找到瓶颈并进行针对性优化。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_training_tips", "content": "神经网络训练技巧包括学习率调整、批归一化、正则化等。训练不收敛时需要检查梯度消失/爆炸问题。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_gradient_problems", "content": "梯度消失和梯度爆炸是深度网络训练的常见问题。使用残差连接和适当的初始化可以缓解这些问题。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_nlp", "content": "自然语言处理（NLP）让机器理解人类语言。现代NLP主要基于深度学习和Transformer架构。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_bert", "content": "BERT是Google开发的双向Transformer编码器，在多项NLP任务上取得突破性成果。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_low_code", "content": "低代码平台允许通过可视化界面和少量代码快速构建应用，降低开发门槛。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_no_code", "content": "无代码平台让非技术人员也能创建应用，完全不需要编写代码，通过拖拽组件实现功能。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_recommendation", "content": "推荐系统根据用户历史行为和偏好推荐商品或内容。协同过滤是经典的推荐算法。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_collaborative_filtering", "content": "协同过滤基于用户行为相似性进行推荐。分为基于用户的协同过滤和基于物品的协同过滤。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_machine_learning", "content": "机器学习让计算机从数据中学习模式。监督学习、无监督学习和强化学习是主要范式。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_clustering", "content": "聚类是无监督学习的重要任务，将数据分成不同的组。K-means和DBSCAN是常用聚类算法。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_security", "content": "数据安全保护信息不被未授权访问。加密、认证和授权是安全的核心机制。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_encryption", "content": "加密将数据转换为不可读形式，只有授权方才能解密。对称加密和非对称加密是两种主要方式。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_distributed_systems", "content": "分布式系统由多台计算机协同工作，提供高可用性和可扩展性。一致性是分布式系统的核心挑战。", "metadata": {"filename": "test_docs.txt"}},
        {"id": "doc_load_balancing", "content": "负载均衡将请求分发到多台服务器，提高系统性能和可用性。常见的负载均衡算法包括轮询和最少连接。", "metadata": {"filename": "test_docs.txt"}},
    ]
    
    print(f"准备导入 {len(test_documents)} 条测试文档...")
    
    # 生成嵌入并添加到向量数据库
    contents = [doc['content'] for doc in test_documents]
    embeddings = embedding_model.embed_texts(contents)
    
    vector_store.add_documents(embeddings, test_documents)
    
    print(f"成功导入 {len(test_documents)} 条测试文档!")
    print(f"当前向量数据库文档数: {vector_store.get_document_count()}")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--import":
        test_import_documents()
    else:
        main()
