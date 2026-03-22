"""
文档摘要生成模块
使用 PyTorch + HuggingFace Transformers 实现
模型: sshleifer/distilbart-cnn-12-6
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from backend.config import settings


class Summarizer:
    """文档摘要生成器 - 单例模式"""
    _instance = None
    _model = None
    _tokenizer = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化摘要模型（仅执行一次）"""
        if Summarizer._model is None:
            model_name = "sshleifer/distilbart-cnn-12-6"
            device = 0 if settings.DEVICE == "cuda" else -1

            # 使用 pipeline 方式加载模型
            Summarizer._pipeline = pipeline(
                "summarization",
                model=model_name,
                device=device,
                torch_dtype=torch.float32
            )

            # 同时保存 tokenizer 和 model 以便精细控制
            Summarizer._tokenizer = AutoTokenizer.from_pretrained(model_name)
            Summarizer._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            Summarizer._model.eval()

            if settings.DEVICE == "cuda":
                Summarizer._model = Summarizer._model.cuda()

            print(f"摘要模型 {model_name} 加载完成")

    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """
        生成文档摘要

        Args:
            text: 输入文本内容
            max_length: 摘要最大长度（token数）
            min_length: 摘要最小长度（token数）

        Returns:
            生成的摘要文本（3句话以内）
        """
        if not text or len(text.strip()) == 0:
            return ""

        # 限制输入长度，避免过长文本
        max_input_length = 1024
        if len(text) > max_input_length * 4:  # 粗略字符估算
            text = text[:max_input_length * 4]

        with torch.no_grad():
            if Summarizer._pipeline is not None:
                # 使用 pipeline 进行摘要
                result = Summarizer._pipeline(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True
                )
                summary = result[0]["summary_text"]
            else:
                # 备用：直接使用 model + tokenizer
                inputs = Summarizer._tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True
                )

                if settings.DEVICE == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                summary_ids = Summarizer._model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )

                summary = Summarizer._tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True
                )

        # 限制为3句话以内
        sentences = summary.split(". ")
        if len(sentences) > 3:
            summary = ". ".join(sentences[:3]) + "."

        return summary.strip()

    def summarize_document(self, text: str) -> str:
        """
        为文档生成简短摘要（对外接口）

        Args:
            text: 文档完整内容

        Returns:
            3句话以内的摘要
        """
        # 清理文本
        text = text.strip().replace("\n", " ").replace("  ", " ")

        # 如果文本太短，直接返回
        if len(text) < 100:
            return text[:200] if text else "文档内容过短，无法生成摘要"

        return self.summarize(text, max_length=120, min_length=20)


# 全局单例实例
_summarizer = None


def get_summarizer() -> Summarizer:
    """获取 Summarizer 单例实例"""
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer


def summarize(text: str) -> str:
    """
    便捷的摘要生成函数

    Args:
        text: 输入文本

    Returns:
        生成的摘要
    """
    summarizer = get_summarizer()
    return summarizer.summarize_document(text)
