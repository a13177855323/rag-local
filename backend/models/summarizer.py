"""
文档摘要生成模块

本模块提供基于 Transformer 的文档摘要功能，使用 DistilBART 模型生成高质量文本摘要。
采用单例模式管理模型实例，实现高效的内存使用和推理性能。

主要特性:
    - 单例模式模型管理，避免重复加载
    - 智能文本分段处理长文档
    - CPU 优化推理，支持 torch.no_grad() 加速
    - 并发安全设计
    - 完善的异常处理和日志记录

使用示例:
    >>> from backend.models.summarizer import get_summarizer, summarize_text
    >>> 
    >>> # 方式1: 使用便捷函数
    >>> summary = summarize_text("这是一段需要摘要的长文本...")
    >>> 
    >>> # 方式2: 使用单例实例
    >>> summarizer = get_summarizer()
    >>> summary = summarizer.summarize("这是一段需要摘要的长文本...")

依赖:
    - torch: PyTorch 深度学习框架
    - transformers: HuggingFace Transformers 库

作者: RAG Development Team
版本: 2.0.0
日期: 2024
"""

import os
import re
import gc
import time
import threading
from typing import List, Optional, Dict, Union
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Pipeline
)

from backend.config import settings


# =============================================================================
# 常量定义
# =============================================================================

# 模型配置常量
DEFAULT_MODEL_NAME: str = "sshleifer/distilbart-cnn-12-6"
"""默认使用的摘要模型名称 (DistilBART-CNN)"""

DEFAULT_MAX_LENGTH: int = 150
"""生成摘要的最大长度 (token 数)"""

DEFAULT_MIN_LENGTH: int = 30
"""生成摘要的最小长度 (token 数)"""

DEFAULT_LENGTH_PENALTY: float = 2.0
"""长度惩罚系数，控制摘要长度偏好"""

DEFAULT_NUM_BEAMS: int = 4
"""束搜索的束宽，影响生成质量和速度"""

DEFAULT_EARLY_STOPPING: bool = True
"""是否启用早停策略"""

# 文本处理常量
MAX_INPUT_LENGTH: int = 1024
"""模型最大输入长度，超过此长度将触发分段处理"""

CHUNK_OVERLAP_RATIO: float = 0.1
"""文本分段时的重叠比例，用于保持上下文连贯性"""

SENTENCE_ENDINGS: str = r'[。！？.!?]'
"""句子结束标记正则表达式"""

# 性能优化常量
CACHE_MAX_SIZE: int = 128
"""LRU 缓存的最大条目数"""

CLEANUP_INTERVAL: int = 100
"""触发垃圾回收的调用间隔"""


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class SummaryConfig:
    """
    摘要生成配置类
    
    用于封装摘要生成的各项参数，便于配置管理和参数传递。
    
    Attributes:
        max_length: 生成摘要的最大 token 数
        min_length: 生成摘要的最小 token 数
        length_penalty: 长度惩罚系数
        num_beams: 束搜索宽度
        early_stopping: 是否启用早停
        do_sample: 是否使用采样策略
        temperature: 采样温度 (仅当 do_sample=True 时有效)
        top_p: 核采样概率阈值 (仅当 do_sample=True 时有效)
    """
    max_length: int = DEFAULT_MAX_LENGTH
    min_length: int = DEFAULT_MIN_LENGTH
    length_penalty: float = DEFAULT_LENGTH_PENALTY
    num_beams: int = DEFAULT_NUM_BEAMS
    early_stopping: bool = DEFAULT_EARLY_STOPPING
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.9
    
    def to_dict(self) -> Dict:
        """将配置转换为字典格式，用于传递给模型"""
        return {
            'max_length': self.max_length,
            'min_length': self.min_length,
            'length_penalty': self.length_penalty,
            'num_beams': self.num_beams,
            'early_stopping': self.early_stopping,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'top_p': self.top_p
        }


@dataclass
class SummaryResult:
    """
    摘要结果数据类
    
    封装摘要生成的结果和元数据。
    
    Attributes:
        summary: 生成的摘要文本
        original_length: 原始文本长度 (字符数)
        summary_length: 摘要长度 (字符数)
        compression_ratio: 压缩比 (摘要长度/原始长度)
        processing_time: 处理耗时 (秒)
        num_chunks: 处理的文本块数量
        model_name: 使用的模型名称
    """
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time: float
    num_chunks: int
    model_name: str
    
    def to_dict(self) -> Dict:
        """将结果转换为字典格式"""
        return {
            'summary': self.summary,
            'original_length': self.original_length,
            'summary_length': self.summary_length,
            'compression_ratio': self.compression_ratio,
            'processing_time': self.processing_time,
            'num_chunks': self.num_chunks,
            'model_name': self.model_name
        }


# =============================================================================
# 异常类定义
# =============================================================================

class SummarizerError(Exception):
    """摘要器基础异常类"""
    pass


class ModelLoadError(SummarizerError):
    """模型加载异常"""
    pass


class TextProcessingError(SummarizerError):
    """文本处理异常"""
    pass


class InferenceError(SummarizerError):
    """推理过程异常"""
    pass


# =============================================================================
# 文本预处理模块
# =============================================================================

class TextPreprocessor:
    """
    文本预处理器
    
    提供文本清洗、分段、长度控制等预处理功能。
    采用静态方法和类方法设计，便于独立使用。
    """
    
    # 编译常用正则表达式，提升性能
    _WHITESPACE_PATTERN = re.compile(r'\s+')
    _SENTENCE_PATTERN = re.compile(r'[^。！？.!?]+[。！？.!?]?')
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        清洗输入文本
        
        移除多余空白字符、特殊符号，规范化文本格式。
        
        Args:
            text: 原始输入文本
            
        Returns:
            清洗后的文本
            
        Raises:
            TextProcessingError: 当输入为空或无效时
        """
        if not text or not isinstance(text, str):
            raise TextProcessingError("输入文本必须是非空字符串")
        
        # 规范化空白字符
        text = cls._WHITESPACE_PATTERN.sub(' ', text)
        
        # 移除控制字符但保留基本标点
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # 去除首尾空白
        text = text.strip()
        
        if not text:
            raise TextProcessingError("清洗后文本为空")
        
        return text
    
    @classmethod
    def split_into_sentences(cls, text: str) -> List[str]:
        """
        将文本分割为句子列表
        
        支持中英文句子边界识别，保留句子结束标记。
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        sentences = cls._SENTENCE_PATTERN.findall(text)
        return [s.strip() for s in sentences if s.strip()]
    
    @classmethod
    def chunk_text(
        cls,
        text: str,
        max_length: int = MAX_INPUT_LENGTH,
        overlap_ratio: float = CHUNK_OVERLAP_RATIO
    ) -> List[str]:
        """
        将长文本分段为适合模型处理的块
        
        采用句子边界感知策略，确保分段不会切断句子。
        相邻块之间保持一定重叠，以维护上下文连贯性。
        
        Args:
            text: 输入文本
            max_length: 每块的最大字符数
            overlap_ratio: 相邻块之间的重叠比例
            
        Returns:
            文本块列表
            
        Raises:
            TextProcessingError: 当分段失败时
        """
        if len(text) <= max_length:
            return [text]
        
        try:
            sentences = cls.split_into_sentences(text)
            if not sentences:
                # 回退策略：按字符长度硬切分
                return cls._fallback_chunk(text, max_length)
            
            chunks = []
            current_chunk = []
            current_length = 0
            overlap_size = int(max_length * overlap_ratio)
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # 单句超过最大长度，需要进一步切分
                if sentence_length > max_length:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # 将长句按字符切分
                    for i in range(0, sentence_length, max_length - overlap_size):
                        chunk = sentence[i:i + max_length]
                        if chunk:
                            chunks.append(chunk)
                    continue
                
                # 检查添加当前句子是否会超出限制
                if current_length + sentence_length + 1 > max_length:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        
                        # 保留部分句子作为重叠
                        overlap_sentences = []
                        overlap_length = 0
                        for s in reversed(current_chunk):
                            if overlap_length + len(s) <= overlap_size:
                                overlap_sentences.insert(0, s)
                                overlap_length += len(s) + 1
                            else:
                                break
                        
                        current_chunk = overlap_sentences
                        current_length = overlap_length
                
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            
            # 添加最后一个块
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks if chunks else [text]
            
        except Exception as e:
            raise TextProcessingError(f"文本分段失败: {str(e)}")
    
    @classmethod
    def _fallback_chunk(cls, text: str, max_length: int) -> List[str]:
        """
        回退分段策略
        
        当句子分割失败时，按固定长度切分文本。
        
        Args:
            text: 输入文本
            max_length: 每块最大长度
            
        Returns:
            文本块列表
        """
        chunks = []
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    @classmethod
    def truncate_text(cls, text: str, max_chars: int = MAX_INPUT_LENGTH) -> str:
        """
        截断文本到指定长度
        
        优先在句子边界处截断，避免切断句子。
        
        Args:
            text: 输入文本
            max_chars: 最大字符数
            
        Returns:
            截断后的文本
        """
        if len(text) <= max_chars:
            return text
        
        # 尝试在句子边界截断
        truncated = text[:max_chars]
        last_sentence_end = max(
            truncated.rfind('。'),
            truncated.rfind('！'),
            truncated.rfind('？'),
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence_end > max_chars * 0.7:  # 至少保留 70% 内容
            return truncated[:last_sentence_end + 1]
        
        return truncated


# =============================================================================
# 模型管理模块
# =============================================================================

class ModelManager:
    """
    模型管理器
    
    负责模型的加载、缓存和内存管理。
    采用单例模式确保全局只有一个模型实例。
    
    Attributes:
        _instance: 单例实例
        _lock: 线程锁，确保并发安全
        _model: 加载的模型对象
        _tokenizer: 加载的分词器对象
        _pipeline: 摘要管道对象
        _model_name: 当前使用的模型名称
        _device: 计算设备 (cpu/cuda)
        _call_count: 调用计数，用于触发垃圾回收
    """
    
    _instance: Optional['ModelManager'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> 'ModelManager':
        """实现线程安全的单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化模型管理器（仅执行一次）"""
        if self._initialized:
            return
        
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._pipeline: Optional[Pipeline] = None
        self._model_name: str = DEFAULT_MODEL_NAME
        self._device: str = settings.DEVICE
        self._call_count: int = 0
        self._initialized = True
    
    def load_model(
        self,
        model_name: Optional[str] = None,
        force_reload: bool = False
    ) -> Pipeline:
        """
        加载摘要模型
        
        使用 transformers pipeline 加载预训练模型，
        支持 CPU 优化和内存管理。
        
        Args:
            model_name: 模型名称，默认使用 DEFAULT_MODEL_NAME
            force_reload: 是否强制重新加载模型
            
        Returns:
            加载好的摘要管道对象
            
        Raises:
            ModelLoadError: 当模型加载失败时
        """
        model_name = model_name or self._model_name
        
        # 检查是否需要重新加载
        if not force_reload and self._pipeline is not None:
            if model_name == self._model_name:
                return self._pipeline
        
        with self._lock:
            try:
                print(f"[Summarizer] 正在加载模型: {model_name}")
                start_time = time.time()
                
                # 清理旧模型以释放内存
                if self._pipeline is not None:
                    self._cleanup_model()
                
                # 设置设备
                device = 0 if self._device == "cuda" and torch.cuda.is_available() else -1
                
                # 加载模型和分词器
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=os.path.join(settings.UPLOAD_DIR, "model_cache")
                )
                
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=os.path.join(settings.UPLOAD_DIR, "model_cache")
                )
                
                # 创建摘要管道
                self._pipeline = pipeline(
                    "summarization",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    device=device,
                    torch_dtype=torch.float32 if device == -1 else torch.float16
                )
                
                self._model_name = model_name
                load_time = time.time() - start_time
                
                print(f"[Summarizer] 模型加载完成，耗时: {load_time:.2f}s")
                
                return self._pipeline
                
            except Exception as e:
                self._cleanup_model()
                raise ModelLoadError(f"模型加载失败: {str(e)}")
    
    def _cleanup_model(self):
        """清理模型资源，释放内存"""
        self._pipeline = None
        self._model = None
        self._tokenizer = None
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_pipeline(self) -> Pipeline:
        """
        获取摘要管道，必要时自动加载模型
        
        Returns:
            摘要管道对象
        """
        if self._pipeline is None:
            return self.load_model()
        return self._pipeline
    
    def get_tokenizer(self) -> AutoTokenizer:
        """
        获取分词器，必要时自动加载
        
        Returns:
            分词器对象
        """
        if self._tokenizer is None:
            self.load_model()
        return self._tokenizer
    
    def increment_call_count(self) -> int:
        """
        增加调用计数
        
        Returns:
            当前计数
        """
        self._call_count += 1
        return self._call_count
    
    def should_cleanup(self) -> bool:
        """
        检查是否应该触发垃圾回收
        
        Returns:
            是否达到清理阈值
        """
        return self._call_count % CLEANUP_INTERVAL == 0


# =============================================================================
# 主摘要器类
# =============================================================================

class Summarizer:
    """
    文档摘要器主类
    
    提供高质量的文档摘要生成功能，支持长文档分段处理、
    批量摘要和性能优化。
    
    采用单例模式管理模型实例，确保内存效率和线程安全。
    
    Attributes:
        _instance: 单例实例
        _lock: 线程锁
        model_manager: 模型管理器实例
        preprocessor: 文本预处理器实例
        config: 默认摘要配置
    
    Example:
        >>> summarizer = Summarizer()
        >>> result = summarizer.summarize("长文本内容...")
        >>> print(result.summary)
    """
    
    _instance: Optional['Summarizer'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> 'Summarizer':
        """实现线程安全的单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        初始化摘要器
        
        Args:
            config: 摘要配置，使用默认配置如果为 None
        """
        if self._initialized:
            return
        
        self.model_manager = ModelManager()
        self.preprocessor = TextPreprocessor()
        self.config = config or SummaryConfig()
        self._initialized = True
    
    def summarize(
        self,
        text: str,
        config: Optional[SummaryConfig] = None
    ) -> SummaryResult:
        """
        生成文本摘要
        
        主入口方法，支持长短文本的自动处理。
        长文本会自动分段处理，然后合并摘要。
        
        Args:
            text: 输入文本
            config: 摘要配置，使用默认配置如果为 None
            
        Returns:
            摘要结果对象，包含摘要文本和元数据
            
        Raises:
            SummarizerError: 当摘要生成失败时
        """
        start_time = time.time()
        config = config or self.config
        
        try:
            # 文本预处理
            cleaned_text = self.preprocessor.clean_text(text)
            original_length = len(cleaned_text)
            
            # 检查是否需要分段
            if original_length > MAX_INPUT_LENGTH:
                return self._summarize_long_text(cleaned_text, config)
            else:
                return self._summarize_short_text(cleaned_text, config, original_length)
                
        except SummarizerError:
            raise
        except Exception as e:
            raise InferenceError(f"摘要生成失败: {str(e)}")
    
    def _summarize_short_text(
        self,
        text: str,
        config: SummaryConfig,
        original_length: int
    ) -> SummaryResult:
        """
        处理短文本摘要
        
        Args:
            text: 清洗后的文本
            config: 摘要配置
            original_length: 原始文本长度
            
        Returns:
            摘要结果
        """
        start_time = time.time()
        
        # 获取模型管道
        pipeline = self.model_manager.get_pipeline()
        
        # 执行推理
        with torch.no_grad():
            result = pipeline(
                text,
                **config.to_dict()
            )
        
        summary = result[0]['summary_text']
        processing_time = time.time() - start_time
        
        # 更新调用计数并检查是否需要清理
        self._update_call_count()
        
        return SummaryResult(
            summary=summary,
            original_length=original_length,
            summary_length=len(summary),
            compression_ratio=len(summary) / original_length if original_length > 0 else 0,
            processing_time=processing_time,
            num_chunks=1,
            model_name=self.model_manager._model_name
        )
    
    def _summarize_long_text(self, text: str, config: SummaryConfig) -> SummaryResult:
        """
        处理长文本摘要（分段处理）
        
        将长文本分段，分别生成摘要，然后合并。
        
        Args:
            text: 清洗后的长文本
            config: 摘要配置
            
        Returns:
            合并后的摘要结果
        """
        start_time = time.time()
        original_length = len(text)
        
        # 分段处理
        chunks = self.preprocessor.chunk_text(text)
        chunk_summaries = []
        
        # 调整每段的摘要长度
        chunk_config = SummaryConfig(
            max_length=min(config.max_length, 100),
            min_length=min(config.min_length, 30),
            length_penalty=config.length_penalty,
            num_beams=config.num_beams,
            early_stopping=config.early_stopping
        )
        
        pipeline = self.model_manager.get_pipeline()
        
        for chunk in chunks:
            with torch.no_grad():
                result = pipeline(
                    chunk,
                    **chunk_config.to_dict()
                )
            chunk_summaries.append(result[0]['summary_text'])
        
        # 合并摘要
        combined_summary = ' '.join(chunk_summaries)
        
        # 如果合并后的摘要仍然太长，进行二次摘要
        if len(chunk_summaries) > 1 and len(combined_summary) > MAX_INPUT_LENGTH:
            combined_summary = self._refine_summary(combined_summary, config)
        
        processing_time = time.time() - start_time
        self._update_call_count()
        
        return SummaryResult(
            summary=combined_summary,
            original_length=original_length,
            summary_length=len(combined_summary),
            compression_ratio=len(combined_summary) / original_length if original_length > 0 else 0,
            processing_time=processing_time,
            num_chunks=len(chunks),
            model_name=self.model_manager._model_name
        )
    
    def _refine_summary(self, summary: str, config: SummaryConfig) -> str:
        """
        精炼摘要（用于长文档的二次摘要）
        
        Args:
            summary: 初步合并的摘要
            config: 摘要配置
            
        Returns:
            精炼后的摘要
        """
        # 截断到合适长度
        truncated = self.preprocessor.truncate_text(summary, MAX_INPUT_LENGTH)
        
        pipeline = self.model_manager.get_pipeline()
        
        refine_config = SummaryConfig(
            max_length=config.max_length,
            min_length=min(config.min_length, 50),
            length_penalty=config.length_penalty,
            num_beams=config.num_beams,
            early_stopping=True
        )
        
        with torch.no_grad():
            result = pipeline(
                truncated,
                **refine_config.to_dict()
            )
        
        return result[0]['summary_text']
    
    def _update_call_count(self):
        """更新调用计数，必要时触发垃圾回收"""
        count = self.model_manager.increment_call_count()
        if count % CLEANUP_INTERVAL == 0:
            gc.collect()
    
    def batch_summarize(
        self,
        texts: List[str],
        config: Optional[SummaryConfig] = None
    ) -> List[SummaryResult]:
        """
        批量生成摘要
        
        Args:
            texts: 文本列表
            config: 摘要配置
            
        Returns:
            摘要结果列表
        """
        results = []
        for text in texts:
            try:
                result = self.summarize(text, config)
                results.append(result)
            except SummarizerError as e:
                # 返回错误结果
                results.append(SummaryResult(
                    summary=f"摘要生成失败: {str(e)}",
                    original_length=len(text) if text else 0,
                    summary_length=0,
                    compression_ratio=0,
                    processing_time=0,
                    num_chunks=0,
                    model_name="error"
                ))
        return results
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_name': self.model_manager._model_name,
            'device': self.model_manager._device,
            'loaded': self.model_manager._pipeline is not None,
            'call_count': self.model_manager._call_count
        }


# =============================================================================
# 便捷函数
# =============================================================================

@lru_cache(maxsize=CACHE_MAX_SIZE)
def _cached_summarize(text_hash: str, text: str, config_json: str) -> str:
    """
    带缓存的摘要生成（内部使用）
    
    使用 LRU 缓存避免重复计算相同文本的摘要。
    
    Args:
        text_hash: 文本哈希值（用于缓存键）
        text: 原始文本
        config_json: 配置 JSON 字符串
        
    Returns:
        摘要文本
    """
    summarizer = Summarizer()
    result = summarizer.summarize(text)
    return result.summary


def summarize_text(
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    min_length: int = DEFAULT_MIN_LENGTH,
    use_cache: bool = True
) -> str:
    """
    便捷函数：生成文本摘要
    
    最简单的使用方式，直接传入文本获取摘要。
    
    Args:
        text: 输入文本
        max_length: 摘要最大长度
        min_length: 摘要最小长度
        use_cache: 是否使用缓存
        
    Returns:
        摘要文本
        
    Example:
        >>> summary = summarize_text("这是一段很长的文本...")
        >>> print(summary)
    """
    if not text or not isinstance(text, str):
        raise ValueError("输入必须是非空字符串")
    
    config = SummaryConfig(max_length=max_length, min_length=min_length)
    
    if use_cache:
        import hashlib
        import json
        text_hash = hashlib.md5(text.encode()).hexdigest()
        config_json = json.dumps(config.to_dict(), sort_keys=True)
        return _cached_summarize(text_hash, text, config_json)
    else:
        summarizer = Summarizer()
        result = summarizer.summarize(text, config)
        return result.summary


def get_summarizer() -> Summarizer:
    """
    获取 Summarizer 单例实例
    
    Returns:
        Summarizer 实例
    """
    return Summarizer()


# =============================================================================
# 模块测试
# =============================================================================

if __name__ == "__main__":
    # 简单测试
    test_text = """
    人工智能（Artificial Intelligence，AI）是指由人制造出来的系统所表现出来的智能。
    人工智能的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    """
    
    print("=" * 50)
    print("文档摘要模块测试")
    print("=" * 50)
    
    # 测试便捷函数
    print("\n1. 测试便捷函数:")
    summary = summarize_text(test_text, max_length=50, min_length=20)
    print(f"摘要: {summary}")
    
    # 测试完整接口
    print("\n2. 测试完整接口:")
    summarizer = get_summarizer()
    result = summarizer.summarize(test_text)
    print(f"摘要: {result.summary}")
    print(f"原始长度: {result.original_length}")
    print(f"摘要长度: {result.summary_length}")
    print(f"压缩比: {result.compression_ratio:.2%}")
    print(f"处理时间: {result.processing_time:.2f}s")
    
    # 测试模型信息
    print("\n3. 模型信息:")
    info = summarizer.get_model_info()
    print(f"模型名称: {info['model_name']}")
    print(f"设备: {info['device']}")
    print(f"已加载: {info['loaded']}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
