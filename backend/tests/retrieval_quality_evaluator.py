#!/usr/bin/env python
"""
RAG检索质量评估脚本 - 独立版本

自动化评估RAG系统的检索性能，支持多种评估指标和批量测试。
支持模拟测试模式，无需实际模型即可运行。

Author: RAG System Test Engineer
Date: 2026-03-24
"""

import os
import sys
import json
import time
import logging
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backend.config import settings
    from backend.services.vector_store import VectorStore
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    print("警告: 无法导入后端模块，将使用模拟模式运行")

try:
    from backend.models.embedding_model import EmbeddingModel
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_UNDERSTANDING = "semantic_understanding"


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TestQuery:
    id: str
    query: str
    query_type: QueryType
    difficulty: DifficultyLevel
    relevant_doc_ids: List[str]
    relevance_scores: Dict[str, int] = field(default_factory=dict)
    description: str = ""


@dataclass
class EvaluationResult:
    query_id: str
    query: str
    query_type: str
    difficulty: str
    retrieved_ids: List[str]
    retrieved_scores: List[float]
    relevant_ids: List[str]
    hit_rate_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_k: Dict[int, float] = field(default_factory=dict)
    response_time_ms: int = 0


@dataclass
class BatchTestResult:
    test_time: str
    total_queries: int
    top_k_values: List[int]
    avg_hit_rate: Dict[int, float] = field(default_factory=dict)
    avg_mrr: float = 0.0
    avg_ndcg: Dict[int, float] = field(default_factory=dict)
    by_query_type: Dict[str, Dict] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    detailed_results: List[EvaluationResult] = field(default_factory=list)


class TestDatasetGenerator:
    def __init__(self):
        self.query_counter = 0
    
    def generate_dataset(self, num_queries: int = 20) -> List[TestQuery]:
        queries = []
        queries_per_type = num_queries // 3
        
        queries.extend(self._generate_exact_match_queries(queries_per_type))
        queries.extend(self._generate_fuzzy_match_queries(queries_per_type))
        queries.extend(self._generate_semantic_queries(num_queries - 2 * queries_per_type))
        
        logger.info(f"生成测试数据集完成，共 {len(queries)} 条查询")
        return queries
    
    def _generate_exact_match_queries(self, count: int) -> List[TestQuery]:
        queries = []
        exact_match_data = [
            ("文档中包含\"机器学习\"的内容是什么？", "机器学习", DifficultyLevel.EASY),
            ("请找出包含\"Python编程\"的段落", "Python编程", DifficultyLevel.EASY),
            ("\"深度学习\"出现在哪些文档中？", "深度学习", DifficultyLevel.MEDIUM),
            ("精确搜索：自然语言处理", "自然语言处理", DifficultyLevel.EASY),
            ("查找关键词：向量数据库", "向量数据库", DifficultyLevel.EASY),
            ("文档中\"RAG系统\"的定义是什么？", "RAG系统", DifficultyLevel.MEDIUM),
            ("搜索\"FAISS索引\"相关内容", "FAISS索引", DifficultyLevel.EASY),
            ("查找包含\"嵌入模型\"的文档", "嵌入模型", DifficultyLevel.EASY),
        ]
        
        for i in range(count):
            data = exact_match_data[i % len(exact_match_data)]
            self.query_counter += 1
            query = TestQuery(
                id=f"exact_{self.query_counter:03d}",
                query=data[0],
                query_type=QueryType.EXACT_MATCH,
                difficulty=data[2],
                relevant_doc_ids=[f"doc_{i % 5}"],
                relevance_scores={f"doc_{i % 5}": 3},
                description=f"精确匹配查询 - 关键词: {data[1]}"
            )
            queries.append(query)
        return queries
    
    def _generate_fuzzy_match_queries(self, count: int) -> List[TestQuery]:
        queries = []
        fuzzy_data = [
            ("关于神经网络的相关信息", "神经网络", DifficultyLevel.MEDIUM),
            ("文本处理方面的内容有哪些？", "文本处理", DifficultyLevel.MEDIUM),
            ("类似Transformer架构的内容", "Transformer", DifficultyLevel.MEDIUM),
            ("跟知识图谱相关的内容", "知识图谱", DifficultyLevel.HARD),
            ("涉及文档检索的文档", "文档检索", DifficultyLevel.MEDIUM),
            ("关于模型训练的相关资料", "模型训练", DifficultyLevel.MEDIUM),
            ("语义相似度计算方法", "语义相似度", DifficultyLevel.HARD),
            ("信息检索技术介绍", "信息检索", DifficultyLevel.MEDIUM),
        ]
        
        for i in range(count):
            data = fuzzy_data[i % len(fuzzy_data)]
            self.query_counter += 1
            relevant_ids = [f"doc_{i % 5}", f"doc_{(i+1) % 5}"]
            query = TestQuery(
                id=f"fuzzy_{self.query_counter:03d}",
                query=data[0],
                query_type=QueryType.FUZZY_MATCH,
                difficulty=data[2],
                relevant_doc_ids=relevant_ids,
                relevance_scores={doc_id: 2 for doc_id in relevant_ids},
                description=f"模糊匹配查询 - 主题: {data[1]}"
            )
            queries.append(query)
        return queries
    
    def _generate_semantic_queries(self, count: int) -> List[TestQuery]:
        queries = []
        semantic_data = [
            ("请解释大语言模型的概念和原理", "大语言模型", DifficultyLevel.HARD),
            ("RAG技术有什么特点和优势？", "RAG技术", DifficultyLevel.MEDIUM),
            ("如何理解向量检索的工作机制？", "向量检索", DifficultyLevel.HARD),
            ("文档分块的应用场景是什么？", "文档分块", DifficultyLevel.MEDIUM),
            ("总结语义搜索的核心要点", "语义搜索", DifficultyLevel.HARD),
            ("比较不同嵌入模型的性能差异", "嵌入模型对比", DifficultyLevel.HARD),
            ("分析检索增强生成的优缺点", "检索增强生成", DifficultyLevel.HARD),
        ]
        
        for i in range(count):
            data = semantic_data[i % len(semantic_data)]
            self.query_counter += 1
            relevant_ids = [f"doc_{j % 5}" for j in range(3)]
            query = TestQuery(
                id=f"semantic_{self.query_counter:03d}",
                query=data[0],
                query_type=QueryType.SEMANTIC_UNDERSTANDING,
                difficulty=data[2],
                relevant_doc_ids=relevant_ids,
                relevance_scores={doc_id: 1 for doc_id in relevant_ids},
                description=f"语义理解查询 - 主题: {data[1]}"
            )
            queries.append(query)
        return queries


class EvaluationMetrics:
    @staticmethod
    def calculate_hit_rate(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        top_k_ids = retrieved_ids[:k]
        hit = any(rid in relevant_ids for rid in top_k_ids)
        return 1.0 if hit else 0.0
    
    @staticmethod
    def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        if not relevant_ids:
            return 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def calculate_ndcg(retrieved_ids: List[str], relevance_scores: Dict[str, int], k: int) -> float:
        def dcg_at_k(relevances: List[int], k: int) -> float:
            dcg = 0.0
            for i, rel in enumerate(relevances[:k]):
                dcg += rel / np.log2(i + 2)
            return dcg
        
        relevances = [relevance_scores.get(rid, 0) for rid in retrieved_ids[:k]]
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        
        dcg = dcg_at_k(relevances, k)
        idcg = dcg_at_k(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
        return dcg / idcg


class MockEmbeddingModel:
    def __init__(self):
        self.dimension = 1024
    
    def embed_query(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


class MockVectorStore:
    def __init__(self):
        self.documents = []
        self._generate_mock_documents()
    
    def _generate_mock_documents(self):
        topics = [
            "机器学习基础", "深度学习原理", "自然语言处理", "Python编程指南",
            "向量数据库技术", "RAG系统架构", "FAISS索引优化", "嵌入模型对比",
            "文档检索方法", "语义搜索技术", "知识图谱构建", "大语言模型应用",
            "Transformer架构", "文档分块策略", "信息检索评估"
        ]
        
        for i, topic in enumerate(topics):
            self.documents.append({
                "id": f"doc_{i}",
                "content": f"{topic}是人工智能领域的重要组成部分。本文档详细介绍了{topic}的核心概念、技术原理和实际应用场景。",
                "metadata": {"filename": f"doc_{i}.txt", "topic": topic}
            })
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        results = []
        for i, doc in enumerate(self.documents[:top_k]):
            np.random.seed(hash(doc["id"]) % (2**32))
            score = random.uniform(0.5, 0.95)
            results.append((doc, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_all_documents(self) -> List[Dict]:
        return self.documents
    
    def get_document_count(self) -> int:
        return len(self.documents)


class RetrievalQualityEvaluator:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock or not HAS_BACKEND or not HAS_EMBEDDING
        
        if self.use_mock:
            self.vector_store = MockVectorStore()
            self.embedding_model = MockEmbeddingModel()
            logger.info("使用模拟模式运行评估")
        else:
            self.vector_store = VectorStore()
            self.embedding_model = EmbeddingModel()
            logger.info("使用实际后端运行评估")
        
        self.dataset_generator = TestDatasetGenerator()
        self.metrics = EvaluationMetrics()
        self.test_queries: List[TestQuery] = []
        self.results: List[EvaluationResult] = []
        self.batch_results: List[BatchTestResult] = []
    
    def generate_test_dataset(self, num_queries: int = 20) -> List[TestQuery]:
        self.test_queries = self.dataset_generator.generate_dataset(num_queries)
        return self.test_queries
    
    def evaluate_single_query(self, query: TestQuery, top_k: int = 10) -> EvaluationResult:
        start_time = time.time()
        
        query_embedding = self.embedding_model.embed_query(query.query)
        search_results = self.vector_store.search(query_embedding, top_k)
        
        response_time = int((time.time() - start_time) * 1000)
        
        retrieved_ids = [doc.get("id", "") for doc, _ in search_results]
        retrieved_scores = [float(score) for _, score in search_results]
        
        hit_rate_k = {}
        ndcg_k = {}
        
        for k in [1, 3, 5, 10]:
            hit_rate_k[k] = self.metrics.calculate_hit_rate(retrieved_ids, query.relevant_doc_ids, k)
            ndcg_k[k] = self.metrics.calculate_ndcg(retrieved_ids, query.relevance_scores, k)
        
        mrr = self.metrics.calculate_mrr(retrieved_ids, query.relevant_doc_ids)
        
        return EvaluationResult(
            query_id=query.id,
            query=query.query,
            query_type=query.query_type.value,
            difficulty=query.difficulty.value,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            relevant_ids=query.relevant_doc_ids,
            hit_rate_k=hit_rate_k,
            mrr=mrr,
            ndcg_k=ndcg_k,
            response_time_ms=response_time
        )
    
    def run_batch_test(self, top_k_values: List[int] = None, queries: List[TestQuery] = None) -> BatchTestResult:
        if top_k_values is None:
            top_k_values = [1, 3, 5, 10]
        
        if queries is None:
            queries = self.test_queries
        
        if not queries:
            logger.error("没有测试查询")
            return BatchTestResult(
                test_time=datetime.now().isoformat(),
                total_queries=0,
                top_k_values=top_k_values
            )
        
        max_k = max(top_k_values)
        detailed_results = []
        
        logger.info(f"开始批量测试，共 {len(queries)} 条查询")
        
        for i, query in enumerate(queries):
            result = self.evaluate_single_query(query, max_k)
            detailed_results.append(result)
            if (i + 1) % 5 == 0:
                logger.info(f"已完成 {i + 1}/{len(queries)} 条查询")
        
        avg_hit_rate = {k: 0.0 for k in top_k_values}
        avg_ndcg = {k: 0.0 for k in top_k_values}
        total_mrr = 0.0
        by_query_type = {}
        by_difficulty = {}
        
        for result in detailed_results:
            for k in top_k_values:
                avg_hit_rate[k] += result.hit_rate_k.get(k, 0)
                avg_ndcg[k] += result.ndcg_k.get(k, 0)
            total_mrr += result.mrr
            
            qt = result.query_type
            if qt not in by_query_type:
                by_query_type[qt] = {'count': 0, 'mrr': 0.0, 'hit_rate': {}, 'ndcg': {}}
            by_query_type[qt]['count'] += 1
            by_query_type[qt]['mrr'] += result.mrr
            for k in top_k_values:
                if k not in by_query_type[qt]['hit_rate']:
                    by_query_type[qt]['hit_rate'][k] = 0.0
                    by_query_type[qt]['ndcg'][k] = 0.0
                by_query_type[qt]['hit_rate'][k] += result.hit_rate_k.get(k, 0)
                by_query_type[qt]['ndcg'][k] += result.ndcg_k.get(k, 0)
            
            diff = result.difficulty
            if diff not in by_difficulty:
                by_difficulty[diff] = {'count': 0, 'mrr': 0.0, 'hit_rate': {}, 'ndcg': {}}
            by_difficulty[diff]['count'] += 1
            by_difficulty[diff]['mrr'] += result.mrr
            for k in top_k_values:
                if k not in by_difficulty[diff]['hit_rate']:
                    by_difficulty[diff]['hit_rate'][k] = 0.0
                    by_difficulty[diff]['ndcg'][k] = 0.0
                by_difficulty[diff]['hit_rate'][k] += result.hit_rate_k.get(k, 0)
                by_difficulty[diff]['ndcg'][k] += result.ndcg_k.get(k, 0)
        
        n = len(detailed_results)
        for k in top_k_values:
            avg_hit_rate[k] /= n
            avg_ndcg[k] /= n
        avg_mrr = total_mrr / n
        
        for qt in by_query_type:
            count = by_query_type[qt]['count']
            by_query_type[qt]['mrr'] /= count
            for k in top_k_values:
                by_query_type[qt]['hit_rate'][k] /= count
                by_query_type[qt]['ndcg'][k] /= count
        
        for diff in by_difficulty:
            count = by_difficulty[diff]['count']
            by_difficulty[diff]['mrr'] /= count
            for k in top_k_values:
                by_difficulty[diff]['hit_rate'][k] /= count
                by_difficulty[diff]['ndcg'][k] /= count
        
        batch_result = BatchTestResult(
            test_time=datetime.now().isoformat(),
            total_queries=n,
            top_k_values=top_k_values,
            avg_hit_rate=avg_hit_rate,
            avg_mrr=avg_mrr,
            avg_ndcg=avg_ndcg,
            by_query_type=by_query_type,
            by_difficulty=by_difficulty,
            detailed_results=detailed_results
        )
        
        self.results = detailed_results
        self.batch_results.append(batch_result)
        return batch_result
    
    def generate_report(self, batch_result: BatchTestResult = None, output_path: str = None) -> str:
        if batch_result is None:
            if not self.batch_results:
                return "没有可用的测试结果"
            batch_result = self.batch_results[-1]
        
        lines = []
        lines.append("=" * 70)
        lines.append("RAG系统检索质量评估报告")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append(f"测试时间: {batch_result.test_time}")
        lines.append(f"测试查询数量: {batch_result.total_queries}")
        lines.append(f"测试top_k值: {batch_result.top_k_values}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("一、整体性能指标")
        lines.append("-" * 70)
        lines.append("")
        
        lines.append("1. Hit Rate@k (前k个结果命中相关文档的比例)")
        lines.append("-" * 40)
        for k in batch_result.top_k_values:
            hr = batch_result.avg_hit_rate.get(k, 0)
            bar = "█" * int(hr * 20)
            lines.append(f"   Hit Rate@{k:2d}: {hr:.4f} {bar}")
        lines.append("")
        
        lines.append("2. MRR (平均倒数排名)")
        lines.append("-" * 40)
        mrr = batch_result.avg_mrr
        bar = "█" * int(mrr * 20)
        lines.append(f"   MRR: {mrr:.4f} {bar}")
        lines.append("")
        
        lines.append("3. NDCG@k (归一化折损累计增益)")
        lines.append("-" * 40)
        for k in batch_result.top_k_values:
            ndcg = batch_result.avg_ndcg.get(k, 0)
            bar = "█" * int(ndcg * 20)
            lines.append(f"   NDCG@{k:2d}: {ndcg:.4f} {bar}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("二、按查询类型分析")
        lines.append("-" * 70)
        lines.append("")
        
        type_names = {
            'exact_match': '精确匹配',
            'fuzzy_match': '模糊匹配',
            'semantic_understanding': '语义理解'
        }
        
        for qt, stats in batch_result.by_query_type.items():
            type_name = type_names.get(qt, qt)
            lines.append(f"【{type_name}】 (共 {stats['count']} 条查询)")
            lines.append(f"   MRR: {stats['mrr']:.4f}")
            for k in batch_result.top_k_values:
                hr = stats['hit_rate'].get(k, 0)
                ndcg = stats['ndcg'].get(k, 0)
                lines.append(f"   Hit Rate@{k}: {hr:.4f}, NDCG@{k}: {ndcg:.4f}")
            lines.append("")
        
        lines.append("-" * 70)
        lines.append("三、按难度级别分析")
        lines.append("-" * 70)
        lines.append("")
        
        diff_names = {'easy': '简单', 'medium': '中等', 'hard': '困难'}
        
        for diff, stats in batch_result.by_difficulty.items():
            diff_name = diff_names.get(diff, diff)
            lines.append(f"【{diff_name}】 (共 {stats['count']} 条查询)")
            lines.append(f"   MRR: {stats['mrr']:.4f}")
            for k in batch_result.top_k_values:
                hr = stats['hit_rate'].get(k, 0)
                ndcg = stats['ndcg'].get(k, 0)
                lines.append(f"   Hit Rate@{k}: {hr:.4f}, NDCG@{k}: {ndcg:.4f}")
            lines.append("")
        
        lines.append("-" * 70)
        lines.append("四、性能评估总结")
        lines.append("-" * 70)
        lines.append("")
        
        hr_5 = batch_result.avg_hit_rate.get(5, 0)
        ndcg_5 = batch_result.avg_ndcg.get(5, 0)
        
        if hr_5 >= 0.8 and ndcg_5 >= 0.7:
            rating = "优秀 ★★★★★"
            suggestion = "检索性能优秀，建议保持当前配置"
        elif hr_5 >= 0.6 and ndcg_5 >= 0.5:
            rating = "良好 ★★★★☆"
            suggestion = "检索性能良好，可考虑优化embedding模型或调整chunk大小"
        elif hr_5 >= 0.4 and ndcg_5 >= 0.3:
            rating = "一般 ★★★☆☆"
            suggestion = "检索性能一般，建议优化文档预处理或增加训练数据"
        else:
            rating = "待改进 ★★☆☆☆"
            suggestion = "检索性能需要改进，建议检查文档质量、调整检索参数"
        
        lines.append(f"综合评级: {rating}")
        lines.append(f"优化建议: {suggestion}")
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("五、详细测试结果")
        lines.append("-" * 70)
        lines.append("")
        
        for result in batch_result.detailed_results[:10]:
            lines.append(f"查询ID: {result.query_id}")
            lines.append(f"   查询: {result.query[:50]}...")
            lines.append(f"   类型: {result.query_type}, 难度: {result.difficulty}")
            lines.append(f"   MRR: {result.mrr:.4f}, 响应时间: {result.response_time_ms}ms")
            lines.append("")
        
        if len(batch_result.detailed_results) > 10:
            lines.append(f"... 省略剩余 {len(batch_result.detailed_results) - 10} 条详细结果")
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("报告结束")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"报告已保存到: {output_path}")
        
        return report


def main():
    print("=" * 70)
    print("RAG系统检索质量评估工具")
    print("=" * 70)
    print()
    
    evaluator = RetrievalQualityEvaluator(use_mock=True)
    
    print("1. 生成测试数据集...")
    test_queries = evaluator.generate_test_dataset(num_queries=20)
    print(f"   生成查询数量: {len(test_queries)}")
    print()
    
    print("2. 运行批量测试...")
    batch_result = evaluator.run_batch_test(top_k_values=[1, 3, 5, 10])
    print()
    
    print("3. 生成评估报告...")
    report = evaluator.generate_report()
    print(report)
    
    output_dir = "./data/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    evaluator.save_results(output_dir) if hasattr(evaluator, 'save_results') else None
    print(f"\n评估完成")


if __name__ == "__main__":
    main()
