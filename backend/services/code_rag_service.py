"""
代码智能检索与问答服务
支持混合检索策略（语义检索 + 代码结构检索）
CPU环境下兼容现有模型适配
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from backend.config import settings
from backend.services.vector_store import VectorStore
from backend.models.embedding_model import EmbeddingModel
from backend.models.llm_model import LLMModel
from backend.utils.code_detector import CodeDetector, get_code_detector, CodeBlock


class CodeRAGService:
    """
    代码RAG服务 - 专门处理代码相关查询
    混合检索策略：语义相似度 + 代码结构匹配 + 关键词匹配
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化服务组件"""
        self.embedding_model = EmbeddingModel()
        self.llm_model = LLMModel()
        self.vector_store = VectorStore()
        self.code_detector = get_code_detector()

        # 检索权重配置
        self.semantic_weight = 0.5
        self.code_structure_weight = 0.3
        self.keyword_weight = 0.2

    def hybrid_search(
        self,
        question: str,
        top_k: int = None,
        code_only: bool = False
    ) -> List[Tuple[Dict, float]]:
        """
        混合检索：结合语义检索和代码结构检索

        Args:
            question: 查询问题
            top_k: 返回结果数量
            code_only: 是否只返回代码块

        Returns:
            检索结果列表 (文档, 综合分数)
        """
        if top_k is None:
            top_k = settings.TOP_K * 2  # 获取更多候选结果

        # 1. 语义检索
        enhanced_query = self.code_detector.enhance_code_query(question)
        query_embedding = self.embedding_model.embed_query(enhanced_query)
        semantic_results = self.vector_store.search(query_embedding, top_k=top_k * 2)

        # 2. 代码结构检索（如果是代码查询）
        if self.code_detector.is_code_query(question):
            code_results = self._code_structure_search(question, top_k=top_k)
            # 合并结果
            combined_results = self._merge_results(semantic_results, code_results, top_k)
        else:
            combined_results = semantic_results[:top_k]

        # 3. 如果只需要代码块，过滤并重新排序
        if code_only:
            combined_results = self._filter_and_rank_code_blocks(
                combined_results, question, top_k
            )

        return combined_results

    def _code_structure_search(
        self,
        question: str,
        top_k: int
    ) -> List[Tuple[Dict, float]]:
        """
        基于代码结构的检索

        Args:
            question: 查询问题
            top_k: 返回结果数量

        Returns:
            代码结构匹配结果
        """
        results = []
        all_docs = self.vector_store.get_all_documents()

        # 提取问题中的关键代码元素
        question_funcs = set(re.findall(r'\b(\w+)\s*\(', question.lower()))
        question_classes = set(re.findall(r'class\s+(\w+)', question.lower()))
        question_keywords = set(re.findall(r'\b\w+\b', question.lower()))

        for doc in all_docs:
            content = doc.get("content", "")
            score = 0.0

            # 提取文档中的代码块
            code_blocks = self.code_detector.extract_code_blocks(content)

            for block in code_blocks:
                # 函数名匹配
                doc_funcs = set(re.findall(r'def\s+(\w+)', block.content))
                func_match = len(question_funcs & doc_funcs)
                score += func_match * 0.3

                # 类名匹配
                doc_classes = set(re.findall(r'class\s+(\w+)', block.content))
                class_match = len(question_classes & doc_classes)
                score += class_match * 0.4

                # 关键词匹配
                doc_keywords = set(re.findall(r'\b\w+\b', block.content.lower()))
                keyword_match = len(question_keywords & doc_keywords)
                score += min(keyword_match * 0.05, 0.3)

                # 代码相关性评分
                relevance = self.code_detector.score_code_relevance(block, question)
                score += relevance * 0.3

            if score > 0:
                results.append((doc, score))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _merge_results(
        self,
        semantic_results: List[Tuple[Dict, float]],
        code_results: List[Tuple[Dict, float]],
        top_k: int
    ) -> List[Tuple[Dict, float]]:
        """
        合并语义检索和代码结构检索结果

        Args:
            semantic_results: 语义检索结果
            code_results: 代码结构检索结果
            top_k: 返回结果数量

        Returns:
            合并后的结果
        """
        # 归一化分数
        semantic_dict = {}
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results)
            for doc, score in semantic_results:
                doc_id = doc.get("id", str(hash(doc.get("content", ""))))
                semantic_dict[doc_id] = (doc, score / max_semantic if max_semantic > 0 else 0)

        code_dict = {}
        if code_results:
            max_code = max(score for _, score in code_results)
            for doc, score in code_results:
                doc_id = doc.get("id", str(hash(doc.get("content", ""))))
                code_dict[doc_id] = (doc, score / max_code if max_code > 0 else 0)

        # 合并分数
        merged = {}
        all_ids = set(semantic_dict.keys()) | set(code_dict.keys())

        for doc_id in all_ids:
            semantic_score = semantic_dict.get(doc_id, (None, 0))[1]
            code_score = code_dict.get(doc_id, (None, 0))[1]

            # 加权融合
            combined_score = (
                self.semantic_weight * semantic_score +
                self.code_structure_weight * code_score
            )

            # 获取文档
            doc = semantic_dict.get(doc_id, code_dict.get(doc_id))[0]
            merged[doc_id] = (doc, combined_score)

        # 排序并返回
        sorted_results = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _filter_and_rank_code_blocks(
        self,
        results: List[Tuple[Dict, float]],
        question: str,
        top_k: int
    ) -> List[Tuple[Dict, float]]:
        """
        过滤并排序代码块

        Args:
            results: 检索结果
            question: 查询问题
            top_k: 返回结果数量

        Returns:
            代码块结果
        """
        code_results = []

        for doc, base_score in results:
            content = doc.get("content", "")
            code_blocks = self.code_detector.extract_code_blocks(content)

            for block in code_blocks:
                # 计算代码块相关性
                relevance = self.code_detector.score_code_relevance(block, question)

                # 综合分数
                final_score = base_score * 0.4 + relevance * 0.6

                # 创建新的文档对象，只包含代码块
                code_doc = doc.copy()
                code_doc["content"] = block.content
                code_doc["metadata"] = {
                    **doc.get("metadata", {}),
                    "is_code_block": True,
                    "language": block.language,
                    "code_context_before": block.context_before,
                    "code_context_after": block.context_after,
                }

                code_results.append((code_doc, final_score))

        # 排序并返回
        code_results.sort(key=lambda x: x[1], reverse=True)
        return code_results[:top_k]

    def query_code(
        self,
        question: str,
        top_k: int = None,
        stream: bool = False
    ) -> Dict or iter:
        """
        代码问答主入口

        Args:
            question: 用户问题
            top_k: 检索结果数量
            stream: 是否流式输出

        Returns:
            回答结果
        """
        if top_k is None:
            top_k = settings.TOP_K

        # 判断是否为代码问题
        is_code = self.code_detector.is_code_query(question)

        # 混合检索
        search_results = self.hybrid_search(
            question,
            top_k=top_k,
            code_only=is_code  # 代码问题只返回代码块
        )

        if not search_results:
            result = {
                "answer": "未找到相关代码，请先上传包含代码的文档。",
                "sources": [],
                "is_code_query": is_code
            }
            return result if not stream else iter([result])

        # 构建上下文
        context_parts = []
        sources = []

        for i, (doc, score) in enumerate(search_results, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # 格式化代码块
            if metadata.get("is_code_block"):
                lang = metadata.get("language", "text")
                context_parts.append(
                    f"【代码片段 {i}】(语言: {lang}, 相似度: {score:.3f})\n"
                    f"```\n{content}\n```"
                )
            else:
                context_parts.append(
                    f"【参考内容 {i}】(相似度: {score:.3f})\n{content}"
                )

            # 构建来源信息
            source = {
                "filename": metadata.get("filename", ""),
                "content": content[:300] + "..." if len(content) > 300 else content,
                "similarity": float(score),
                "is_code": metadata.get("is_code_block", False),
                "language": metadata.get("language", None)
            }
            sources.append(source)

        context = "\n\n".join(context_parts)

        # 构建提示词
        if is_code:
            prompt = self._build_code_prompt(question, context)
        else:
            prompt = self._build_general_prompt(question, context)

        # 生成回答
        if stream:
            return self._stream_response(prompt, sources, is_code)
        else:
            answer = self.llm_model.generate(prompt, [context])
            formatted_answer = self._format_code_answer(answer, is_code)

            return {
                "answer": formatted_answer,
                "sources": sources,
                "is_code_query": is_code
            }

    def _build_code_prompt(self, question: str, context: str) -> str:
        """构建代码问答提示词"""
        return f"""你是一个专业的代码助手。请根据提供的代码片段回答用户的问题。

用户问题：{question}

相关代码片段：
{context}

请提供：
1. 直接回答用户的问题
2. 如果涉及代码修改或示例，请提供完整的代码
3. 解释关键代码逻辑
4. 如果适用，提供使用示例

回答格式：
- 使用Markdown代码块展示代码
- 代码中保留原有注释
- 对关键行添加解释"""

    def _build_general_prompt(self, question: str, context: str) -> str:
        """构建通用问答提示词"""
        return f"""基于以下参考内容回答问题：

参考内容：
{context}

问题：{question}

请提供准确、简洁的回答。"""

    def _stream_response(self, prompt: str, sources: List[Dict], is_code: bool):
        """流式响应生成器"""
        full_response = ""
        for chunk in self.llm_model.generate_stream(prompt, []):
            full_response += chunk
            yield {
                "answer": chunk,
                "done": False,
                "sources": sources,
                "is_code_query": is_code
            }

        # 格式化最终回答
        formatted = self._format_code_answer(full_response, is_code)

        yield {
            "answer": "",
            "done": True,
            "sources": sources,
            "is_code_query": is_code,
            "formatted_answer": formatted
        }

    def _format_code_answer(self, answer: str, is_code: bool) -> str:
        """
        格式化代码回答

        Args:
            answer: 原始回答
            is_code: 是否为代码问题

        Returns:
            格式化后的回答
        """
        if not is_code:
            return answer

        # 确保代码块使用正确的Markdown格式
        formatted = answer

        # 修复不规范的代码块标记
        formatted = re.sub(r'```\s*\n\s*```', '', formatted)  # 删除空代码块
        formatted = re.sub(r'`{3,}', '```', formatted)  # 统一代码块标记

        # 确保Python代码有语言标记
        formatted = re.sub(
            r'```\s*\n(def\s+|class\s+|import\s+)',
            r'```python\n\1',
            formatted
        )

        return formatted

    def extract_code_from_answer(self, answer: str) -> List[Dict]:
        """
        从回答中提取代码块

        Args:
            answer: 回答文本

        Returns:
            代码块列表
        """
        code_blocks = []
        pattern = r'```(\w*)\n(.*?)```'

        for match in re.finditer(pattern, answer, re.DOTALL):
            lang = match.group(1) or "text"
            code = match.group(2).strip()

            code_blocks.append({
                "language": lang,
                "code": code,
                "length": len(code)
            })

        return code_blocks


# 全局单例实例
_code_rag_service = None


def get_code_rag_service() -> CodeRAGService:
    """获取 CodeRAGService 单例实例"""
    global _code_rag_service
    if _code_rag_service is None:
        _code_rag_service = CodeRAGService()
    return _code_rag_service
