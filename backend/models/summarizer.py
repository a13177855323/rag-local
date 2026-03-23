import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from backend.config import settings


class Summarizer:
    """文档摘要生成器，使用单例模式"""
    _instance = None
    _model_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化模型（仅加载一次）"""
        if self._model_loaded:
            return

        model_name = "sshleifer/distilbart-cnn-12-6"
        self.device = settings.DEVICE if torch.cuda.is_available() else "cpu"

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

        # 创建pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device if self.device == "cpu" else 0
        )

        self._model_loaded = True

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """
        生成文档摘要

        Args:
            text: 输入文本
            max_sentences: 最大句子数（默认3句）

        Returns:
            生成的摘要文本
        """
        if not text or len(text.strip()) == 0:
            return ""

        # 限制输入长度（BART支持的最大长度）
        max_input_length = 1024
        inputs = self.tokenizer(
            text,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # 推理时禁用梯度计算
        with torch.no_grad():
            # 生成摘要
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # 确保摘要不超过max_sentences句
        sentences = [s.strip() for s in summary.split(". ") if s.strip()]
        if len(sentences) > max_sentences:
            summary = ". ".join(sentences[:max_sentences]) + "."

        return summary

    def summarize_long(self, text: str, max_sentences: int = 3) -> str:
        """
        处理长文本摘要（当文本超过模型最大输入长度时）

        Args:
            text: 输入文本
            max_sentences: 最大句子数（默认3句）

        Returns:
            生成的摘要文本
        """
        if not text or len(text.strip()) == 0:
            return ""

        # 使用pipeline处理长文本，会自动分块
        result = self.summarizer(
            text,
            max_length=150,
            min_length=40,
            do_sample=False
        )

        summary = result[0]["summary_text"]

        # 确保摘要不超过max_sentences句
        sentences = [s.strip() for s in summary.split(". ") if s.strip()]
        if len(sentences) > max_sentences:
            summary = ". ".join(sentences[:max_sentences]) + "."

        return summary
