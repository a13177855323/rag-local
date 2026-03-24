#!/usr/bin/env python3
"""
RAG系统检索质量评估脚本

功能：
1. 构造测试数据集：生成20条不同难度级别的测试查询
2. 实现评估指标计算：Hit Rate@k、MRR、NDCG
3. 批量测试：支持批量导入测试文档，自动记录不同top_k参数下的检索性能
4. 结果可视化：生成性能对比报告

约束：
- 复用VectorStore和EmbeddingModel组件
- CPU环境高效运行
- 兼容FAISS索引结构
- backend/tests目录
"""
# 解决macOS上FAISS和PyTorch的OpenMP库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import sys
sys.path.insert(0, '.')

import json
import tempfile
import shutil
from typing import List, Dict, Tuple, Any
from datetime import datetime
from collections import defaultdict

import numpy as np

# 注意：必须先加载嵌入模型（PyTorch），再初始化FAISS，否则会有库冲突
from backend.models.embedding_model import EmbeddingModel
from backend.services.vector_store import VectorStore
from backend.config import settings

class RetrievalEvaluator:
    """
    检索质量评估器
    
    负责：
    1. 准备测试文档和查询
    2. 执行检索测试
    3. 计算评估指标
    4. 生成评估报告
    """
    
    def __init__(self, test_dir: str = None):
        """
        初始化评估器
        
        Args:
            test_dir: 测试数据目录，None则使用临时目录
        """
        self.test_dir = test_dir or tempfile.mkdtemp(prefix="rag_test_")
        self.embedding_model = EmbeddingModel()
        self.vector_store = None
        self.test_queries = []
        self.test_documents = []
        self.query_relevant_docs = defaultdict(list)  # query -> relevant_doc_ids
        
    def _init_test_vector_store(self) -> VectorStore:
        """初始化测试用的向量存储"""
        # 创建临时向量库
        original_path = settings.VECTOR_DB_PATH
        settings.VECTOR_DB_PATH = os.path.join(self.test_dir, "vector_store")
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        # 重新初始化VectorStore单例
        VectorStore._instance = None
        self.vector_store = VectorStore()
        return self.vector_store
        
    def generate_test_documents(self) -> List[Dict[str, Any]]:
        """
        生成测试文档集
        
        返回：
            测试文档列表，包含不同主题：
            - 技术文档（Python、FAISS等
            - 系统文档（Linux命令、架构）
            - 自然语言处理文档（向量数据库、RAG）
            - API文档
        """
        test_docs = [
            # Python编程相关
            {"content": "Python是一种高级编程语言，以简洁著称，支持面向对象、函数式等多种编程范式。Python的语法设计哲学强调代码的可读性。", "doc_id": "py_001", "category": "python"},
            {"content": "Python的列表推导式（List Comprehension）是一种简洁的创建列表的语法，例如：[x for x in range(10)]", "doc_id": "py_002", "category": "python"},
            {"content": "Python中的装饰器（Decorator）可以在不修改原函数代码的情况下扩展函数功能，常见的装饰器有@staticmethod、@classmethod等。", "doc_id": "py_003", "category": "python"},
            {"content": "Python的异常处理使用try-except-finally结构，可以捕获和处理程序运行时的错误和异常情况。", "doc_id": "py_004", "category": "python"},
            {"content": "Python中的生成器（Generator）使用yield关键字，能够实现惰性计算，节省内存空间。", "doc_id": "py_005", "category": "python"},
            
            # FAISS向量数据库
            {"content": "FAISS是Facebook开发的向量数据库，支持高效的相似性搜索和聚类。FAISS支持的索引类型有IndexFlatL2、IndexIVFFlat等。", "doc_id": "fais_001", "category": "faiss"},
            {"content": "FAISS的IndexHNSWFlat索引采用层次化导航搜索图（HNSW）结构，能够在高维空间中实现高效的近似最近邻搜索。", "doc_id": "fais_002", "category": "faiss"},
            {"content": "FAISS支持GPU加速，可以显著提高大规模向量检索的速度和效率。", "doc_id": "fais_003", "category": "faiss"},
            {"content": "向量数据库专门用于存储和检索高维向量数据，是RAG系统中的核心组件是检索与召回，相似度计算通常使用余弦相似度、L2距离等度量方式。", "doc_id": "vec_001", "category": "vector_db"},
            {"content": "RAG系统即检索增强生成（Retrieval-Augmented Generation）结合了检索和生成两个阶段，能够利用外部知识库来提高大语言模型的回答准确性。", "doc_id": "rag_001", "category": "rag"},
            
            # Linux系统
            {"content": "Linux是一种开源操作系统，常用命令有ls、cd、pwd、grep、find等。Linux的文件系统采用树状结构组织。", "doc_id": "linux_001", "category": "linux"},
            {"content": "Linux中的grep命令用于在文件中搜索指定的模式，支持正则表达式匹配。", "doc_id": "linux_002", "category": "linux"},
            {"content": "Linux文件权限分为读、写、执行三种，分别用r、w、x表示，可以通过chmod命令修改。", "doc_id": "linux_003", "category": "linux"},
            {"content": "Shell脚本是Linux/Unix系统中的脚本编程语言，可以自动化执行一系列命令，支持变量、循环、条件判断等功能。", "doc_id": "linux_004", "category": "linux"},
            
            # API与Web
            {"content": "RESTful API是一种软件架构风格，使用HTTP方法如GET、POST、PUT、DELETE等操作资源。RESTful API通常返回JSON格式数据。", "doc_id": "api_001", "category": "api"},
            {"content": "HTTP协议的状态码：200表示成功，404表示未找到，500表示服务器内部错误。", "doc_id": "api_002", "category": "api"},
            {"content": "FastAPI是一个现代、快速的Python Web框架，基于标准Python类型提示自动生成API文档。", "doc_id": "api_003", "category": "api"},
            
            # 算法与数据结构
            {"content": "快速排序是一种高效的排序算法，采用分治策略，平均时间复杂度O(n log n)。", "doc_id": "algo_001", "category": "algorithm"},
            {"content": "二叉树是每个节点最多有两个子树的树状数据结构，常用于实现快速查找。", "doc_id": "algo_002", "category": "algorithm"},
        ]
        
        # 标准格式转换
        formatted_docs = []
        for i, doc in enumerate(test_docs):
            formatted_docs.append({
                "content": doc["content"],
                "metadata": {
                    "doc_id": doc["doc_id"],
                    "category": doc["category"],
                    "filename": f"test_doc_{i:03d}.txt"
                }
            })
        
        self.test_documents = formatted_docs
        return formatted_docs
    
    def generate_test_queries(self) -> List[Dict[str, Any]]:
        """
        生成20条测试查询，包含三类：
        1. 精确匹配查询（6条）：关键词精确出现在文档中
        2. 模糊匹配查询（7条）：关键词部分匹配或同义词
        3. 语义理解查询（7条）：需要理解语义才能匹配
        
        返回：
            测试查询列表，包含查询内容、类型、相关文档ID列表
        """
        queries = [
            # ===== 精确匹配查询（6条）
            {
                "query_id": "q_exact_001",
                "query_text": "Python中的装饰器是什么",
                "type": "exact",
                "difficulty": "easy",
                "relevant_docs": ["py_003"],
                "keywords": ["Python装饰器"]
            },
            {
                "query_id": "q_exact_002",
                "query_text": "FAISS支持哪些索引类型",
                "type": "exact",
                "difficulty": "easy",
                "relevant_docs": ["fais_001", "fais_002"],
                "keywords": ["FAISS索引"]
            },
            {
                "query_id": "q_exact_003",
                "query_text": "Linux grep命令用法",
                "type": "exact",
                "difficulty": "easy",
                "relevant_docs": ["linux_002"],
                "keywords": ["Linux grep"]
            },
            {
                "query_id": "q_exact_004",
                "query_text": "RESTful API有哪些HTTP方法",
                "type": "exact",
                "difficulty": "easy",
                "relevant_docs": ["api_001"],
                "keywords": ["RESTful HTTP方法"]
            },
            {
                "query_id": "q_exact_005",
                "query_text": "Python列表推导式语法",
                "type": "exact",
                "difficulty": "easy",
                "relevant_docs": ["py_002"],
                "keywords": ["Python列表推导式"]
            },
            {
                "query_id": "q_exact_006",
                "query_text": "快速排序算法时间复杂度",
                "type": "exact",
                "difficulty": "easy",
                "relevant_docs": ["algo_001"],
                "keywords": ["快速排序", "时间复杂度"]
            },
            
            # ===== 模糊匹配查询（7条）
            {
                "query_id": "q_fuzzy_001",
                "query_text": "怎样用Python写一个装饰器",
                "type": "fuzzy",
                "difficulty": "medium",
                "relevant_docs": ["py_003"],
                "keywords": ["Python装饰器写法"]
            },
            {
                "query_id": "q_fuzzy_002",
                "query_text": "Facebook的向量检索工具有哪些",
                "type": "fuzzy",
                "difficulty": "medium",
                "relevant_docs": ["fais_001", "fais_002", "fais_003"],
                "keywords": ["Facebook向量检索工具"]
            },
            {
                "query_id": "q_fuzzy_003",
                "query_text": "在Linux中搜索文件内容",
                "type": "fuzzy",
                "difficulty": "medium",
                "relevant_docs": ["linux_002"],
                "keywords": ["Linux文件内容搜索"]
            },
            {
                "query_id": "q_fuzzy_004",
                "query_text": "什么是REST架构",
                "type": "fuzzy",
                "difficulty": "medium",
                "relevant_docs": ["api_001"],
                "keywords": ["REST架构", "REST API"]
            },
            {
                "query_id": "q_fuzzy_005",
                "query_text": "Python生成器用法",
                "type": "fuzzy",
                "difficulty": "medium",
                "relevant_docs": ["py_005"],
                "keywords": ["Python yield生成器"]
            },
            {
                "query_id": "q_fuzzy_006",
                "query_text": "向量数据库作用是什么",
                "type": "fuzzy",
                "difficulty": "medium",
                "relevant_docs": ["vec_001"],
                "keywords": ["向量数据库作用"]
            },
            {
                "query_id": "q_fuzzy_007",
                "query_text": "Linux文件权限设置",
                "type": "fuzzy",
                "difficulty": "medium",
                "relevant_docs": ["linux_003"],
                "keywords": ["Linux权限设置"]
            },
            
            # ===== 语义理解查询（7条）
            {
                "query_id": "q_semantic_001",
                "query_text": "我想了解RAG技术",
                "type": "semantic",
                "difficulty": "hard",
                "relevant_docs": ["rag_001"],
                "keywords": ["RAG技术原理"]
            },
            {
                "query_id": "q_semantic_002",
                "query_text": "如何加快检索外部知识优化大模型回答",
                "type": "semantic",
                "difficulty": "hard",
                "relevant_docs": ["rag_001"],
                "keywords": ["RAG系统"]
            },
            {
                "query_id": "q_semantic_003",
                "query_text": "Web开发用什么框架好",
                "type": "semantic",
                "difficulty": "hard",
                "relevant_docs": ["api_003"],
                "keywords": ["Web开发框架FastAPI"]
            },
            {
                "query_id": "q_semantic_004",
                "query_text": "数据结构中二叉树是什么",
                "type": "semantic",
                "difficulty": "hard",
                "relevant_docs": ["algo_002"],
                "keywords": ["二叉树应用场景"]
            },
            {
                "query_id": "q_semantic_005",
                "query_text": "怎样实现惰性计算",
                "type": "semantic",
                "difficulty": "hard",
                "relevant_docs": ["py_005"],
                "keywords": ["Python中节省内存"]
            },
            {
                "query_id": "q_semantic_006",
                "query_text": "服务器遇到错误怎么办",
                "type": "semantic",
                "difficulty": "hard",
                "relevant_docs": ["py_004"],
                "keywords": ["错误处理", "异常"]
            },
            {
                "query_id": "q_semantic_007",
                "query_text": "高性能相似性搜索技术",
                "type": "semantic",
                "difficulty": "hard",
                "relevant_docs": ["fais_001"],
                "keywords": ["相似性搜索技术"]
            }
        ]
        
        # 构建查询到相关文档的映射
        for q in queries:
            self.query_relevant_docs[q["query_id"]] = q["relevant_docs"]
        
        self.test_queries = queries
        print(f"生成测试查询：{len(queries)}条，其中：")
        print(f"  精确匹配：{sum(1 for q in queries if q['type'] == 'exact')}条")
        print(f"  模糊匹配：{sum(1 for q in queries if q['type'] == 'fuzzy')}条")
        print(f"  语义理解：{sum(1 for q in queries if q['type'] == 'semantic')}条")
        
        return queries
    
    def index_test_documents(self) -> bool:
        """将测试文档索引到向量数据库"""
        if not self.test_documents:
            self.generate_test_documents()
            
        # 初始化向量库
        self._init_test_vector_store()
        
        # 生成嵌入并索引
        texts = [doc["content"] for doc in self.test_documents]
        embeddings = self.embedding_model.embed_documents(texts)
        
        # 添加到向量库
        self.vector_store.add_documents(embeddings, self.test_documents)
        
        print(f"索引测试文档：{len(self.test_documents)}条")
        return True
    
    def _get_doc_id(self, result: Tuple[Dict, float]) -> str:
        """从检索结果中提取文档ID"""
        metadata = result[0].get("metadata", {})
        return metadata.get("doc_id", "")
    
    def calculate_hit_rate_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        计算Hit Rate@k：前k个结果中是否有相关文档
        
        参数：
            retrieved_docs: 检索到的文档ID列表
            relevant_docs: 真实相关文档ID列表
            k: 取前k个
            
        返回：
            1表示命中，0表示未命中
        """
        retrieved_top_k = retrieved_docs[:k]
        return 1.0 if any(doc_id in relevant_docs for doc_id in retrieved_top_k) else 0.0
    
    def calculate_mrr(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        计算MRR（平均倒数排名：第一个相关文档的排名的倒数。如果没有相关文档，返回0。
        """
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / i
        return 0.0
    
    def calculate_ndcg(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        计算NDCG（归一化折损累计增益）
        
        参数：
            retrieved_docs: 检索到的文档ID列表
            relevant_docs: 真实相关文档ID列表
            
        返回：
            NDCG分数(0-1)
        """
        if not relevant_docs:
            return 0.0
            
        # 计算DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(i + 1)
        
        # 计算IDCG（理想DCG，即相关文档都在最前面的情况）
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, len(relevant_docs) + 1))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def run_evaluation(
        self,
        top_k_list: List[int] = None
    ) -> Dict[str, Any]:
        """
        运行检索评估测试
        
        参数：
            top_k_list: 测试的top_k参数列表，默认[1, 3, 5, 10]
            
        返回：
            评估结果
        """
        if top_k_list is None:
            top_k_list = [1, 3, 5, 10]
            
        if not self.test_queries:
            self.generate_test_queries()
            
        if self.vector_store is None:
            self.index_test_documents()
            
        results = {
            "summary": {},
            "by_query": [],
            "by_top_k": {},
            "by_type": {}
        }
        
        # 按top_k分组统计
        for top_k in top_k_list:
            results["by_top_k"][str(top_k)] = {
                "hit_rate": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
                "queries": []
            }
        
        # 按查询类型分组
        query_types = ["exact", "fuzzy", "semantic"]
        for qtype in query_types:
            results["by_type"][qtype] = {
                "hit_rate": defaultdict(float),
                "mrr": defaultdict(float),
                "count": 0
            }
        
        # 执行每个查询
        total_queries = []
        for query_info in self.test_queries:
            query_id = query_info["query_id"]
            query_text = query_info["query_text"]
            relevant_docs = query_info["relevant_docs"]
            query_type = query_info["type"]
            
            # 生成查询嵌入
            query_embedding = self.embedding_model.embed_query(query_text)
            
            # 使用最大的top_k进行检索，之后在计算指标时再截断
            max_k = max(top_k_list)
            search_results = self.vector_store.search(query_embedding, top_k=max_k)
            
            # 提取文档ID
            retrieved_docs = [self._get_doc_id(res) for res in search_results]
            
            query_result = {
                "query_id": query_id,
                "query_text": query_text,
                "query_type": query_type,
                "retrieved_docs": retrieved_docs[:max_k],
                "relevant_docs": relevant_docs,
                "metrics": {}
            }
            
            # 计算不同top_k下的指标
            for top_k in top_k_list:
                metrics = {}
                metrics["hit_rate"] = self.calculate_hit_rate_at_k(
                    retrieved_docs, relevant_docs, top_k
                )
                metrics["mrr"] = self.calculate_mrr(
                    retrieved_docs[:top_k], relevant_docs
                )
                metrics["ndcg"] = self.calculate_ndcg(
                    retrieved_docs[:top_k], relevant_docs
                )
                
                query_result["metrics"][str(top_k)] = metrics
                
                # 累加到top_k分组
                results["by_top_k"][str(top_k)]["hit_rate"] += metrics["hit_rate"]
                results["by_top_k"][str(top_k)]["mrr"] += metrics["mrr"]
                results["by_top_k"][str(top_k)]["ndcg"] += metrics["ndcg"]
                
                # 累加到类型分组
                results["by_type"][query_type]["hit_rate"][top_k] += metrics["hit_rate"]
                results["by_type"][query_type]["mrr"][top_k] += metrics["mrr"]
            
            results["by_query"].append(query_result)
            results["by_type"][query_type]["count"] += 1
        
        # 计算平均值
        n_queries = len(self.test_queries)
        
        for top_k in top_k_list:
            results["by_top_k"][str(top_k)]["hit_rate"] /= n_queries
            results["by_top_k"][str(top_k)]["mrr"] /= n_queries
            results["by_top_k"][str(top_k)]["ndcg"] /= n_queries
            
            # 删除每个查询的详细结果
            results["by_top_k"][str(top_k)]["queries"] = [
                q for q in results["by_query"]
            ]
        
        # 计算各类型的平均
        for qtype in query_types:
            count = results["by_type"][qtype]["count"]
            if count > 0:
                for top_k in top_k_list:
                    results["by_type"][qtype]["hit_rate"][top_k] /= count
                    results["by_type"][qtype]["mrr"][top_k] /= count
        
        # 总体摘要
        results["summary"] = {
            "total_queries": n_queries,
            "total_documents": len(self.test_documents),
            "test_types": {
                "exact": sum(1 for q in self.test_queries if q["type"] == "exact"),
                "fuzzy": sum(1 for q in self.test_queries if q["type"] == "fuzzy"),
                "semantic": sum(1 for q in self.test_queries if q["type"] == "semantic")
            },
            "top_k_list": top_k_list,
            "overall_performance": {}
        }
        
        # 总体性能（取top_k=5作为代表）
        if 5 in top_k_list:
            results["summary"]["overall_performance"] = {
                "hit_rate@5": results["by_top_k"]["5"]["hit_rate"],
                "mrr@5": results["by_top_k"]["5"]["mrr"],
                "ndcg@5": results["by_top_k"]["5"]["ndcg"]
            }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成文本格式的性能对比报告
        
        参数：
            results: 评估结果字典
            
        返回：
            格式化的报告字符串
        """
        report = []
        report.append("=" * 80)
        report.append("RAG系统检索质量评估报告")
        report.append("=" * 80)
        report.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试文档数: {results['summary']['total_documents']}")
        report.append(f"测试查询数: {results['summary']['total_queries']} (精确: {results['summary']['test_types']['exact']}, 模糊: {results['summary']['test_types']['fuzzy']}, 语义: {results['summary']['test_types']['semantic']})")
        
        report.append("\n" + "-" * 80)
        report.append("一、整体性能表现")
        report.append("-" * 80)
        
        # 按top_k的性能表
        headers = ["指标"] + [f"top_{k}" for k in results["summary"]["top_k_list"]]
        report.append(f"{'指标':<15} {'top_1':<10} {'top_3':<10} {'top_5':<10} {'top_10':<10}")
        report.append("-" * 60)
        
        for metric in ["hit_rate", "mrr", "ndcg"]:
            row = [f"{metric.upper():<15}"]
            for k in results["summary"]["top_k_list"]:
                value = results["by_top_k"][str(k)][metric]
                row.append(f"{value:.4f}")
            report.append(" ".join(f"{v:<10}" for v in row))
        
        report.append("\n" + "-" * 80)
        report.append("二、按查询类型的性能分析（top_k=5)")
        report.append("-" * 80)
        
        type_names = {
            "exact": "精确匹配",
            "fuzzy": "模糊匹配",
            "semantic": "语义理解"
        }
        
        report.append(f"{'查询类型':<15} {'HitRate@5':<12} {'MRR@5':<12}")
        report.append("-" * 45)
        
        for qtype, type_name in type_names.items():
            hr = results["by_type"][qtype]["hit_rate"][5]
            mrr = results["by_type"][qtype]["mrr"][5]
            count = results["by_type"][qtype]["count"]
            type_display = f"{type_name}({count}条)"
            report.append(f"{type_display:<15} {hr:<12.4f} {mrr:<12.4f}")
        
        report.append("\n" + "-" * 80)
        report.append("三、查询详细结果（前5条示例）")
        report.append("-" * 80)
        
        for i, query_result in enumerate(results["by_query"][:5]):
            report.append(f"\n查询{i+1}: {query_result['query_text'][:50]}...")
            report.append(f"  类型: {type_names.get(query_result['query_type'])}")
            report.append(f"  相关文档数: {len(query_result['relevant_docs'])}")
            
            # 显示top_5结果:
            top5_docs = query_result['retrieved_docs'][:5]
            report.append(f"  检索到前5文档: {', '.join(top5_docs)}")
            
            # 是否命中:
            hr5 = query_result['metrics']['5']['hit_rate']
            mrr5 = query_result['metrics']['5']['mrr']
            report.append(f"  性能: HitRate@5={hr5:.2f}, MRR@5={mrr5:.4f}")
        
        report.append("\n" + "=" * 80)
        report.append("四、测试结论")
        report.append("=" * 80)
        
        overall = results["summary"]["overall_performance"]
        report.append(f"总体 HitRate@5: {overall.get('hit_rate@5', 0):.2%}")
        report.append(f"总体 MRR@5: {overall.get('mrr@5', 0):.4f}")
        report.append(f"总体 NDCG@5: {overall.get('ndcg@5', 0):.4f}")
        
        return "\n".join(report)
    
    def save_results_to_file(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        保存结果到JSON文件
        
        参数：
            results: 评估结果字典
            filename: 输出文件名，None则自动生成
            
        返回：
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"retrieval_evaluation_{timestamp}.json"
        
        filepath = os.path.join(self.test_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {filepath}")
        return filepath
    
    def cleanup(self):
        """清理临时测试数据"""
        # 恢复原设置
        from backend.config import settings
        settings.VECTOR_DB_PATH = "./data/vector_store"
        
        # 删除临时目录
        if os.path.exists(self.test_dir) and self.test_dir.startswith("/tmp"):
            shutil.rmtree(self.test_dir, ignore_errors=True)
            print(f"清理临时目录: {self.test_dir}")


def main():
    """主函数：执行评估测试"""
    print("=" * 80)
    print("RAG系统检索质量评估工具")
    print("=" * 80)
    
    # 创建评估器
    evaluator = RetrievalEvaluator()
    
    try:
        # 1. 准备测试数据
        print("\n[1/4] 生成测试文档...")
        evaluator.generate_test_documents()
        
        print("\n[2/4] 生成测试查询...")
        evaluator.generate_test_queries()
        
        # 2. 索引文档
        print("\n[3/4] 索引测试文档到向量数据库...")
        evaluator.index_test_documents()
        
        # 3. 运行评估
        print("\n[4/4] 运行检索评估...")
        results = evaluator.run_evaluation(top_k_list=[1, 3, 5, 10])
        
        # 4. 生成报告
        print("\n" + "=" * 80)
        report = evaluator.generate_report(results)
        print(report)
        
        # 保存结果
        evaluator.save_results_to_file(results)
        
    finally:
        evaluator.cleanup()
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()
