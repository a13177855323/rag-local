"""
模型模块

提供文档摘要、嵌入向量生成、语言模型等核心模型组件。
"""

from backend.models.summarizer import (
    DocumentSummarizer,
    SummarizerConfig,
    SummarizerError,
    ModelLoadError,
    TokenizerLoadError,
    SummarizationError,
    get_summarizer,
    summarize,
    summarize_documents,
)

__all__ = [
    "DocumentSummarizer",
    "SummarizerConfig",
    "SummarizerError",
    "ModelLoadError",
    "TokenizerLoadError",
    "SummarizationError",
    "get_summarizer",
    "summarize",
    "summarize_documents",
]
