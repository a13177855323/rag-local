import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional
from backend.config import settings


class SummarizerModel:
    _instance: Optional["SummarizerModel"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.model_name = "sshleifer/distilbart-cnn-12-6"
        self.device = settings.DEVICE
        self._model = None
        self._tokenizer = None
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._pipeline = pipeline(
                "summarization",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.device == "cuda" else -1
            )

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        if not text or not text.strip():
            return ""

        self._load_model()

        max_input_length = 1024
        if len(text.split()) > max_input_length:
            text = " ".join(text.split()[:max_input_length])

        with torch.no_grad():
            result = self._pipeline(
                text,
                max_length=150,
                min_length=30,
                do_sample=False,
                truncation=True
            )

        summary = result[0]["summary_text"]

        sentences = self._split_sentences(summary)
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
            summary = " ".join(sentences)

        return summary

    def _split_sentences(self, text: str) -> list:
        import re
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences


_summarizer_instance: Optional[SummarizerModel] = None


def get_summarizer() -> SummarizerModel:
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = SummarizerModel()
    return _summarizer_instance


def summarize(text: str, max_sentences: int = 3) -> str:
    return get_summarizer().summarize(text, max_sentences)
