# encoding: utf-8
"""
文本预处理器单元测试
测试TextPreprocessor类的各项功能
"""

import unittest
import sys
import os

# 添加项目路径到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.services.conversation_analyzer import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    """TextPreprocessor单元测试类"""

    def test_clean_text_basic(self):
        """测试基础文本清洗功能"""
        # 测试去除多余空格
        text = "  这是   一个   测试  文本  "
        cleaned = TextPreprocessor.clean_text(text)
        self.assertEqual(cleaned, "这是 一个 测试 文本")

        # 测试去除特殊空白字符
        text = "这是\u3000一个\xa0测试"
        cleaned = TextPreprocessor.clean_text(text)
        self.assertEqual(cleaned, "这是 一个 测试")

        # 测试空文本
        text = ""
        cleaned = TextPreprocessor.clean_text(text)
        self.assertEqual(cleaned, "")

    def test_clean_text_newlines(self):
        """测试换行符处理"""
        text = "第一行\n第二行\r\n第三行\r"
        cleaned = TextPreprocessor.clean_text(text)
        self.assertEqual(cleaned, "第一行\n第二行\n第三行")

        # 测试合并多个空行
        text = "第一段\n\n\n第二段\n\n第三段"
        cleaned = TextPreprocessor.clean_text(text)
        self.assertEqual(cleaned, "第一段\n第二段\n第三段")

    def test_split_sentences(self):
        """测试分句功能"""
        # 中文分句测试
        text = "这是第一句。这是第二句！这是第三句？"
        sentences = TextPreprocessor.split_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "这是第一句。")
        self.assertEqual(sentences[1], "这是第二句！")
        self.assertEqual(sentences[2], "这是第三句")

        # 英文分句测试
        text = "Hello! How are you? I'm fine."
        sentences = TextPreprocessor.split_sentences(text)
        self.assertEqual(len(sentences), 3)

        # 空文本测试
        sentences = TextPreprocessor.split_sentences("")
        self.assertEqual(sentences, [])

    def test_tokenize(self):
        """测试分词功能"""
        text = "这是一个测试文本"
        tokens = TextPreprocessor.tokenize(text, remove_stopwords=False)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # 测试去停用词
        text = "这是一个测试的文本"
        tokens_with_stopwords = TextPreprocessor.tokenize(text, remove_stopwords=False)
        tokens_without_stopwords = TextPreprocessor.tokenize(text, remove_stopwords=True)
        self.assertLessEqual(len(tokens_without_stopwords), len(tokens_with_stopwords))

    def test_extract_keywords(self):
        """测试关键词提取功能"""
        text = """
        人工智能是计算机科学的一个分支，它企图了解智能的实质，
        并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        """
        keywords = TextPreprocessor.extract_keywords(text, top_n=5)
        self.assertIsInstance(keywords, list)
        self.assertEqual(len(keywords), 5)
        # 应该包含"智能"或"人工智能"等关键词
        has_ai = any('智能' in kw or '人工智能' in kw for kw in keywords)
        self.assertTrue(has_ai)

    def test_calculate_similarity(self):
        """测试文本相似度计算"""
        text1 = "人工智能在医疗领域的应用"
        text2 = "人工智能在医疗行业的应用"
        text3 = "机器学习在金融领域的应用"

        # 相似文本应该有较高的相似度
        sim1 = TextPreprocessor.calculate_similarity(text1, text2)
        sim2 = TextPreprocessor.calculate_similarity(text1, text3)
        self.assertGreater(sim1, sim2)

        # 相同文本的相似度
        sim_identical = TextPreprocessor.calculate_similarity(text1, text1)
        self.assertGreater(sim_identical, 0.8)

        # 空文本相似度
        sim_empty = TextPreprocessor.calculate_similarity("", text1)
        self.assertEqual(sim_empty, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
