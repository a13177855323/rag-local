"""
文档摘要生成模块

提供基于预训练模型的文档自动摘要功能。
使用 sshleifer/distilbart-cnn-12-6 模型生成新闻类文档摘要。
支持单例模式加载模型，优化CPU环境下的资源占用。

Author: RAG System
Date: 2026-03-23
"""

import logging
import os
import re
from typing import List, Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from backend.config import settings


# 配置日志格式：时间戳 - 错误等级 - [模块标识] - 详细描述
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class SummarizerError(Exception):
    """摘要生成器相关异常基类"""
    pass


class ModelLoadError(SummarizerError):
    """模型加载失败异常"""
    pass


class TokenizerLoadError(SummarizerError):
    """Tokenizer加载失败异常"""
    pass


class SummarizationError(SummarizerError):
    """摘要生成失败异常"""
    pass


class SummarizerConfig:
    """
    摘要生成器配置管理类

    管理模型配置、生成参数和性能优化选项。

    Attributes:
        MODEL_NAME: 预训练模型名称
        DEVICE: 运行设备 (cpu/cuda)
        MAX_INPUT_LENGTH: 输入文本最大token数
        MAX_OUTPUT_LENGTH: 输出摘要最大token数
        MIN_OUTPUT_LENGTH: 输出摘要最小token数
        DEFAULT_MAX_SENTENCES: 默认摘要句子数量
    """

    MODEL_NAME: str = "sshleifer/distilbart-cnn-12-6"
    DEVICE: str = "cpu"
    MAX_INPUT_LENGTH: int = 1024
    MAX_OUTPUT_LENGTH: int = 150
    MIN_OUTPUT_LENGTH: int = 30
    DEFAULT_MAX_SENTENCES: int = 3

    # CPU优化配置
    CPU_OPTIMIZATION = {
        "num_threads": 4,
        "enable_fp16": False,
        "lazy_load": True,
    }

    @classmethod
    def from_settings(cls) -> "SummarizerConfig":
        """
        从全局配置创建设置实例

        Returns:
            SummarizerConfig 实例
        """
        config = cls()
        config.DEVICE = getattr(settings, 'DEVICE', 'cpu')
        return config


class SentenceSplitter:
    """
    句子切分工具类

    支持中英文混合文本的句子边界识别和切分。
    """

    # 句子结束标点符号
    SENTENCE_DELIMITERS = [
        '.', '!', '?', '。', '！', '？',
    ]

    @classmethod
    def split(cls, text: str) -> List[str]:
        """
        切分文本为句子列表

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        if not text:
            return []

        # 使用正则表达式匹配句子边界
        pattern = r'[.!?。！？]+'
        sentences = re.split(pattern, text)

        # 过滤空句子并去除首尾空白
        sentences = [
            s.strip()
            for s in sentences
            if s.strip()
        ]

        return sentences

    @classmethod
    def truncate_to_sentences(
        cls,
        text: str,
        max_sentences: int
    ) -> str:
        """
        截取指定数量的句子

        Args:
            text: 输入文本
            max_sentences: 最大句子数

        Returns:
            截断后的文本
        """
        sentences = cls.split(text)
        if len(sentences) <= max_sentences:
            return text

        return ' '.join(sentences[:max_sentences])


class TextPreprocessor:
    """
    文本预处理工具类

    负责输入文本的清洗、截断和规范化。
    """

    # 空白字符正则
    WHITESPACE_PATTERN = re.compile(r'\s+')

    @classmethod
    def clean(cls, text: str) -> str:
        """
        清洗文本：去除多余空白字符

        Args:
            text: 输入文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # 替换多余空白为单个空格
        text = cls.WHITESPACE_PATTERN.sub(' ', text)

        return text.strip()

    @classmethod
    def truncate_by_tokens(
        cls,
        text: str,
        max_words: int
    ) -> str:
        """
        按单词数截断文本

        Args:
            text: 输入文本
            max_words: 最大单词数

        Returns:
            截断后的文本
        """
        if not text:
            return ""

        words = text.split()
        if len(words) <= max_words:
            return text

        return ' '.join(words[:max_words])


class ModelLoader:
    """
    模型加载器类

    负责模型的懒加载、缓存和设备配置。
    采用单例模式确保模型只加载一次。

    Attributes:
        model_name: 预训练模型名称
        device: 运行设备
        model: 加载的模型实例
        tokenizer: 加载的tokenizer实例
        summarization_pipeline: 摘要生成管道
    """

    def __init__(self, config: SummarizerConfig):
        """
        初始化模型加载器

        Args:
            config: 摘要生成器配置
        """
        self.config = config
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._is_loaded = False

    def _get_device_id(self) -> int:
        """
        获取设备ID

        Returns:
            设备ID (cuda: 0, cpu: -1)
        """
        if self.config.DEVICE == "cuda" and torch.cuda.is_available():
            return 0
        return -1

    def load(self) -> None:
        """
        加载模型和tokenizer

        使用懒加载机制，仅在首次调用时加载模型。
        包含异常捕获和日志记录。

        Raises:
            ModelLoadError: 模型加载失败时抛出
            TokenizerLoadError: Tokenizer加载失败时抛出
        """
        if self._is_loaded:
            logger.debug("模型已加载，跳过")
            return

        try:
            logger.info(f"开始加载模型: {self.config.MODEL_NAME}")

            # 加载tokenizer
            self._tokenizer = self._load_tokenizer()

            # 加载模型
            self._model = self._load_model()

            # 移动到指定设备
            self._model.to(self.config.DEVICE)

            # 设置评估模式
            self._model.eval()

            # 创建摘要管道
            self._pipeline = self._create_pipeline()

            self._is_loaded = True
            logger.info(f"模型加载完成，设备: {self.config.DEVICE}")

        except OSError as e:
            error_msg = f"模型文件不存在或损坏: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        except Exception as e:
            error_msg = f"模型加载失败: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)

    def _load_tokenizer(self):
        """
        加载Tokenizer

        Returns:
            Tokenizer实例

        Raises:
            TokenizerLoadError: 加载失败时抛出
        """
        try:
            logger.debug("加载Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.MODEL_NAME
            )
            logger.debug("Tokenizer加载成功")
            return tokenizer
        except Exception as e:
            error_msg = f"Tokenizer加载失败: {e}"
            logger.error(error_msg)
            raise TokenizerLoadError(error_msg)

    def _load_model(self):
        """
        加载预训练模型

        Returns:
            模型实例
        """
        logger.debug("加载模型权重...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.MODEL_NAME
        )

        # CPU优化配置
        if self.config.DEVICE == "cpu":
            if hasattr(torch, 'set_num_threads'):
                threads = self.config.CPU_OPTIMIZATION.get('num_threads', 4)
                torch.set_num_threads(threads)
                logger.debug(f"设置CPU线程数: {threads}")

        return model

    def _create_pipeline(self):
        """
        创建摘要生成管道

        Returns:
            摘要pipeline实例
        """
        logger.debug("创建摘要生成管道...")
        return pipeline(
            "summarization",
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._get_device_id(),
        )

    @property
    def pipeline(self):
        """
        获取摘要管道

        如果模型未加载，先执行加载。

        Returns:
            摘要pipeline实例
        """
        if not self._is_loaded:
            self.load()
        return self._pipeline

    def unload(self) -> None:
        """
        卸载模型释放内存

        用于内存资源紧张时的手动清理。
        """
        if self._is_loaded:
            del self._pipeline
            del self._model
            del self._tokenizer
            self._pipeline = None
            self._model = None
            self._tokenizer = None
            self._is_loaded = False
            logger.info("模型已卸载，内存已释放")

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DocumentSummarizer:
    """
    文档摘要生成器主类

    整合模型加载、文本处理和摘要生成功能。
    采用单例模式确保全局唯一实例。

    Example:
        >>> summarizer = DocumentSummarizer()
        >>> summary = summarizer.summarize("这是一个很长的文档...")
        >>> print(summary)
    """

    _instance: Optional["DocumentSummarizer"] = None

    def __new__(cls, config: Optional[SummarizerConfig] = None):
        """
        单例模式创建实例

        Args:
            config: 可选的配置实例

        Returns:
            DocumentSummarizer实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(config)
        return cls._instance

    def _initialize(self, config: Optional[SummarizerConfig] = None) -> None:
        """
        初始化摘要生成器

        Args:
            config: 配置实例，默认使用环境配置
        """
        self.config = config or SummarizerConfig.from_settings()
        self.model_loader = ModelLoader(self.config)
        self.preprocessor = TextPreprocessor()
        self.splitter = SentenceSplitter()

        logger.info("文档摘要生成器初始化完成")

    def summarize(
        self,
        text: str,
        max_sentences: int = None,
        max_length: int = None,
        min_length: int = None,
    ) -> str:
        """
        生成文档摘要

        Args:
            text: 输入文档文本
            max_sentences: 摘要最大句子数 (默认3)
            max_length: 生成摘要最大token数
            min_length: 生成摘要最小token数

        Returns:
            生成的摘要文本

        Raises:
            SummarizationError: 摘要生成失败时抛出
        """
        # 参数校验
        if not text or not text.strip():
            logger.warning("输入文本为空，返回空摘要")
            return ""

        # 使用默认值
        max_sentences = max_sentences or self.config.DEFAULT_MAX_SENTENCES
        max_length = max_length or self.config.MAX_OUTPUT_LENGTH
        min_length = min_length or self.config.MIN_OUTPUT_LENGTH

        try:
            # 文本预处理
            cleaned_text = self.preprocessor.clean(text)
            truncated_text = self.preprocessor.truncate_by_tokens(
                cleaned_text,
                self.config.MAX_INPUT_LENGTH
            )

            # 使用torch.no_grad()禁用梯度计算，优化推理性能
            with torch.no_grad():
                result = self.model_loader.pipeline(
                    truncated_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True,
                )

            # 提取摘要文本
            summary = result[0]["summary_text"]

            # 按句子数截断
            final_summary = self.splitter.truncate_to_sentences(
                summary,
                max_sentences
            )

            logger.debug(f"摘要生成成功，原文长度: {len(text)}, 摘要长度: {len(final_summary)}")
            return final_summary

        except Exception as e:
            error_msg = f"摘要生成失败: {e}"
            logger.error(error_msg)
            raise SummarizationError(error_msg)

    def summarize_batch(
        self,
        texts: List[str],
        max_sentences: int = None,
    ) -> List[str]:
        """
        批量生成文档摘要

        Args:
            texts: 输入文档文本列表
            max_sentences: 摘要最大句子数

        Returns:
            摘要文本列表

        Example:
            >>> texts = ["文档1内容...", "文档2内容..."]
            >>> summaries = summarizer.summarize_batch(texts)
        """
        summaries = []
        for text in texts:
            try:
                summary = self.summarize(
                    text,
                    max_sentences=max_sentences,
                )
                summaries.append(summary)
            except SummarizationError as e:
                logger.error(f"批量摘要生成跳过失败文档: {e}")
                summaries.append("")

        return summaries


# ======================== 模块级便捷函数 ========================

# 全局单例实例
_summarizer_instance: Optional[DocumentSummarizer] = None


def get_summarizer(config: Optional[SummarizerConfig] = None) -> DocumentSummarizer:
    """
    获取文档摘要生成器单例实例

    Args:
        config: 可选的配置实例

    Returns:
        DocumentSummarizer实例
    """
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = DocumentSummarizer(config)
    return _summarizer_instance


def summarize(
    text: str,
    max_sentences: int = 3,
) -> str:
    """
    生成文档摘要的便捷函数

    Args:
        text: 输入文档文本
        max_sentences: 摘要最大句子数 (默认3)

    Returns:
        生成的摘要文本

    Example:
        >>> summary = summarize("这是一个很长的文档内容...")
        >>> print(summary)
    """
    return get_summarizer().summarize(text, max_sentences=max_sentences)


def summarize_documents(
    documents: List[str],
    max_sentences: int = 3,
) -> List[str]:
    """
    批量生成文档摘要的便捷函数

    Args:
        documents: 文档文本列表
        max_sentences: 摘要最大句子数

    Returns:
        摘要列表

    Example:
        >>> docs = ["文档1...", "文档2..."]
        >>> summaries = summarize_documents(docs)
    """
    return get_summarizer().summarize_batch(
        documents,
        max_sentences=max_sentences,
    )
