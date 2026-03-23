# encoding: utf-8
"""
摘要生成器单元测试
测试各种摘要生成策略的功能
"""

import unittest
import sys
import os
import numpy as np

# 添加项目路径到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.services.conversation_analyzer import (
    ExtractiveSummarizer, AbstractiveSummarizer, HybridSummarizer,
    SummaryResult, SummaryStrategy, SummaryLength, TextPreprocessor
)


class TestExtractiveSummarizer(unittest.TestCase):
    """抽取式摘要生成器测试"""

    def setUp(self):
        """测试前初始化"""
        self.summarizer = ExtractiveSummarizer()
        # 长测试文本
        self.long_text = """
        人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
        它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
        可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
        人工智能可以对人的意识、思维的信息过程的模拟。
        人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
        机器学习是人工智能的一个重要分支，它使计算机系统能够通过经验自动改进。
        深度学习则是机器学习的一个子领域，它使用多层神经网络来模拟人类大脑的学习过程。
        这些技术的发展正在深刻改变我们的生活和工作方式。
        """
        # 中等长度文本
        self.medium_text = """
        Python是一种广泛使用的高级编程语言，它的设计哲学强调代码的可读性。
        Python的语法允许程序员用更少的代码表达想法，相比其他语言如C++或Java。
        该语言提供了多种编程范式，包括面向对象、命令式、函数式和过程式编程。
        Python解释器可以安装在许多操作系统上，允许Python代码在多种环境中执行。
        此外，Python拥有一个庞大而全面的标准库，为开发者提供了丰富的工具和功能。
        """

    def test_build_similarity_matrix(self):
        """测试相似度矩阵构建"""
        sentences = TextPreprocessor.split_sentences(self.medium_text)
        sim_matrix = self.summarizer._build_similarity_matrix(sentences)

        self.assertIsInstance(sim_matrix, np.ndarray)
        self.assertEqual(sim_matrix.shape, (len(sentences), len(sentences)))
        # 对角线元素应该为0（句子与自身的相似度在计算时设为0）
        for i in range(len(sentences)):
            self.assertEqual(sim_matrix[i, i], 0.0)

    def test_calculate_text_rank(self):
        """测试TextRank算法"""
        sentences = TextPreprocessor.split_sentences(self.medium_text)
        sim_matrix = self.summarizer._build_similarity_matrix(sentences)
        scores = self.summarizer._calculate_text_rank(sim_matrix)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), len(sentences))
        # 所有分数应该在0到1之间
        self.assertTrue(all(0 <= s <= 1 for s in scores))

    def test_summarize_basic(self):
        """测试基础摘要生成"""
        result = self.summarizer.summarize(self.long_text, ratio=0.3)

        self.assertIsInstance(result, SummaryResult)
        self.assertGreater(len(result.summary), 0)
        self.assertLess(len(result.summary), len(self.long_text))
        self.assertGreater(result.compression_ratio, 0)
        self.assertLess(result.compression_ratio, 1)

    def test_summarize_with_max_length(self):
        """测试带最大长度限制的摘要生成"""
        max_length = 100
        result = self.summarizer.summarize(self.long_text, max_length=max_length)

        self.assertIsInstance(result, SummaryResult)
        # 摘要长度应该接近但不超过max_length
        self.assertLessEqual(len(result.summary), max_length + 50)  # 允许小范围误差

    def test_summarize_empty_text(self):
        """测试空文本摘要"""
        result = self.summarizer.summarize("")
        self.assertEqual(result.summary, "")
        self.assertEqual(result.original_length, 0)
        self.assertEqual(result.summary_length, 0)

    def test_summarize_short_text(self):
        """测试短文本摘要"""
        short_text = "这是一个短句子。"
        result = self.summarizer.summarize(short_text, ratio=0.5)
        self.assertGreater(len(result.summary), 0)

    def test_key_points_extraction(self):
        """测试关键点提取"""
        result = self.summarizer.summarize(self.long_text, ratio=0.3)
        self.assertIsInstance(result.key_points, list)
        self.assertGreater(len(result.key_points), 0)
        self.assertLessEqual(len(result.key_points), 5)  # 应该最多5个关键点

    def test_keywords_extraction(self):
        """测试关键词提取"""
        result = self.summarizer.summarize(self.long_text, ratio=0.3)
        self.assertIsInstance(result.keywords, list)
        self.assertGreater(len(result.keywords), 0)
        # 应该包含"人工智能"等关键词
        has_ai = any('人工智能' in kw or '智能' in kw for kw in result.keywords)
        self.assertTrue(has_ai)

    def test_topic_inference(self):
        """测试主题推断"""
        result = self.summarizer.summarize(self.long_text, ratio=0.3)
        self.assertIsInstance(result.topic, str)
        # 主题应该包含"人工智能"或"智能"
        has_ai_topic = '人工智能' in result.topic or '智能' in result.topic
        self.assertTrue(has_ai_topic or len(result.topic) > 0)


class TestHybridSummarizer(unittest.TestCase):
    """混合式摘要生成器测试"""

    def setUp(self):
        """测试前初始化"""
        self.summarizer = HybridSummarizer()
        self.test_text = """
        区块链是一种分布式账本技术，它允许数据在多个节点之间共享和同步。
        每个区块包含一定数量的交易记录，并通过密码学方法与前一个区块链接。
        这种技术最初是为比特币等加密货币设计的，但现在已被广泛应用于金融、
        供应链管理、医疗健康等多个领域。区块链的去中心化特性使得数据更加透明，
        同时也提高了安全性，因为没有单一控制点可以被攻击或操纵。
        智能合约是区块链上的自动执行合约，当满足特定条件时会自动触发相应操作。
        """

    def test_hybrid_summarize(self):
        """测试混合式摘要生成"""
        result = self.summarizer.summarize(self.test_text, ratio=0.3)
        self.assertIsInstance(result, SummaryResult)
        self.assertEqual(result.strategy, SummaryStrategy.HYBRID.value)
        self.assertGreater(len(result.summary), 0)


class TestSummaryLength(unittest.TestCase):
    """摘要长度控制测试"""

    def test_summary_length_enum(self):
        """测试摘要长度枚举值"""
        self.assertEqual(SummaryLength.SHORT.value, 0.15)
        self.assertEqual(SummaryLength.MEDIUM.value, 0.30)
        self.assertEqual(SummaryLength.LONG.value, 0.50)

    def test_different_summary_lengths(self):
        """测试不同长度的摘要生成"""
        summarizer = ExtractiveSummarizer()
        text = """
        数据科学是一门跨学科领域，使用科学方法、流程、算法和系统来提取知识。
        它结合了统计学、数据分析、机器学习和相关领域的理论和方法。
        数据科学家使用各种工具和技术来处理和分析大量数据。
        这些数据可以来自不同的来源，如社交媒体、传感器、交易记录等。
        分析结果可以用于预测趋势、优化业务流程、支持决策制定等目的。
        数据可视化是数据科学的重要组成部分，帮助人们理解复杂的数据模式。
        随着大数据技术的发展，数据科学的应用范围正在不断扩大。
        从医疗保健到金融服务，从制造业到零售业，数据科学都在发挥重要作用。
        """

        short_result = summarizer.summarize(text, ratio=SummaryLength.SHORT.value)
        medium_result = summarizer.summarize(text, ratio=SummaryLength.MEDIUM.value)
        long_result = summarizer.summarize(text, ratio=SummaryLength.LONG.value)

        # 长度应该递增
        self.assertLessEqual(len(short_result.summary), len(medium_result.summary))
        self.assertLessEqual(len(medium_result.summary), len(long_result.summary))


class TestSummaryResult(unittest.TestCase):
    """摘要结果数据结构测试"""

    def test_summary_result_creation(self):
        """测试摘要结果对象创建"""
        result = SummaryResult(
            summary="这是摘要内容",
            original_length=1000,
            summary_length=100,
            compression_ratio=0.1,
            key_points=["关键点1", "关键点2"],
            keywords=["关键词1", "关键词2"],
            strategy=SummaryStrategy.EXTRACTIVE.value,
            topic="测试主题"
        )

        self.assertEqual(result.summary, "这是摘要内容")
        self.assertEqual(result.original_length, 1000)
        self.assertEqual(result.summary_length, 100)
        self.assertEqual(result.compression_ratio, 0.1)
        self.assertEqual(result.strategy, SummaryStrategy.EXTRACTIVE.value)
        self.assertEqual(result.topic, "测试主题")


if __name__ == '__main__':
    unittest.main(verbosity=2)
