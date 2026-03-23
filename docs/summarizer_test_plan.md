# 文档摘要功能 - 功能验证方案

## 1. 概述

本文档提供文档摘要功能的全面验证方案，包括单元测试、性能基准测试、功能对比测试及边界条件测试，确保重构后的代码在保持原有功能的同时实现了预期的质量提升。

---

## 2. 测试目标

### 2.1 功能验证
- 验证摘要生成的准确性和完整性
- 验证长短文本的处理能力
- 验证批量处理功能

### 2.2 性能验证
- 验证模型加载时间和内存占用
- 验证推理速度和吞吐量
- 验证缓存机制的有效性

### 2.3 稳定性验证
- 验证异常处理能力
- 验证并发安全性
- 验证内存管理效果

---

## 3. 测试环境

### 3.1 硬件要求
- CPU: 4核以上
- 内存: 8GB+
- 磁盘: SSD 推荐

### 3.2 软件环境
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- pytest 7.0+

---

## 4. 单元测试

### 4.1 TextPreprocessor 测试

```python
# tests/test_text_preprocessor.py

class TestTextPreprocessor:
    """文本预处理器测试类"""
    
    def test_clean_text_normal(self):
        """测试正常文本清洗"""
        input_text = "  这是   一段\t需要\n清洗的  文本  "
        expected = "这是 一段 需要 清洗的 文本"
        result = TextPreprocessor.clean_text(input_text)
        assert result == expected
    
    def test_clean_text_empty(self):
        """测试空文本异常"""
        with pytest.raises(TextProcessingError):
            TextPreprocessor.clean_text("")
    
    def test_clean_text_none(self):
        """测试 None 输入异常"""
        with pytest.raises(TextProcessingError):
            TextPreprocessor.clean_text(None)
    
    def test_split_into_sentences_chinese(self):
        """测试中文句子分割"""
        text = "这是第一句。这是第二句！这是第三句？"
        sentences = TextPreprocessor.split_into_sentences(text)
        assert len(sentences) == 3
    
    def test_split_into_sentences_english(self):
        """测试英文句子分割"""
        text = "This is first. This is second! This is third?"
        sentences = TextPreprocessor.split_into_sentences(text)
        assert len(sentences) == 3
    
    def test_chunk_text_short(self):
        """测试短文本不分段"""
        text = "这是一段短文本。"
        chunks = TextPreprocessor.chunk_text(text, max_length=1000)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_long(self):
        """测试长文本分段"""
        # 生成长文本
        text = "这是一段测试文本。" * 1000
        chunks = TextPreprocessor.chunk_text(text, max_length=500)
        assert len(chunks) > 1
        # 验证每段不超过限制
        for chunk in chunks:
            assert len(chunk) <= 500 * 1.2  # 允许 20% 溢出
    
    def test_truncate_text(self):
        """测试文本截断"""
        text = "这是第一句。这是第二句。这是第三句。"
        result = TextPreprocessor.truncate_text(text, max_chars=15)
        assert len(result) <= 15
        # 验证在句子边界截断
        assert result.endswith('。')
```

### 4.2 ModelManager 测试

```python
# tests/test_model_manager.py

class TestModelManager:
    """模型管理器测试类"""
    
    def test_singleton(self):
        """测试单例模式"""
        mm1 = ModelManager()
        mm2 = ModelManager()
        assert mm1 is mm2
    
    def test_load_model(self):
        """测试模型加载"""
        mm = ModelManager()
        pipeline = mm.load_model()
        assert pipeline is not None
        assert mm._pipeline is not None
    
    def test_load_model_cache(self):
        """测试模型缓存"""
        mm = ModelManager()
        pipeline1 = mm.load_model()
        pipeline2 = mm.load_model()
        assert pipeline1 is pipeline2  # 应该返回同一实例
    
    def test_force_reload(self):
        """测试强制重新加载"""
        mm = ModelManager()
        pipeline1 = mm.load_model()
        pipeline2 = mm.load_model(force_reload=True)
        # 强制重新加载后应该是新实例
        assert pipeline1 is not pipeline2
    
    def test_cleanup_model(self):
        """测试模型清理"""
        mm = ModelManager()
        mm.load_model()
        mm._cleanup_model()
        assert mm._pipeline is None
        assert mm._model is None
    
    def test_increment_call_count(self):
        """测试调用计数"""
        mm = ModelManager()
        initial = mm._call_count
        mm.increment_call_count()
        assert mm._call_count == initial + 1
    
    def test_should_cleanup(self):
        """测试清理判断"""
        mm = ModelManager()
        mm._call_count = CLEANUP_INTERVAL
        assert mm.should_cleanup() is True
        mm._call_count = CLEANUP_INTERVAL - 1
        assert mm.should_cleanup() is False
```

### 4.3 Summarizer 测试

```python
# tests/test_summarizer.py

class TestSummarizer:
    """摘要器主类测试"""
    
    def test_singleton(self):
        """测试单例模式"""
        s1 = Summarizer()
        s2 = Summarizer()
        assert s1 is s2
    
    def test_summarize_short_text(self):
        """测试短文本摘要"""
        summarizer = Summarizer()
        text = "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。"
        result = summarizer.summarize(text)
        assert isinstance(result, SummaryResult)
        assert len(result.summary) > 0
        assert result.num_chunks == 1
    
    def test_summarize_long_text(self):
        """测试长文本摘要"""
        summarizer = Summarizer()
        # 生成长文本 (> MAX_INPUT_LENGTH)
        text = "人工智能是计算机科学的一个分支。" * 1000
        result = summarizer.summarize(text)
        assert isinstance(result, SummaryResult)
        assert result.num_chunks > 1  # 应该分段处理
    
    def test_summarize_empty_text(self):
        """测试空文本异常"""
        summarizer = Summarizer()
        with pytest.raises(SummarizerError):
            summarizer.summarize("")
    
    def test_summarize_invalid_text(self):
        """测试无效输入异常"""
        summarizer = Summarizer()
        with pytest.raises(SummarizerError):
            summarizer.summarize(None)
    
    def test_batch_summarize(self):
        """测试批量摘要"""
        summarizer = Summarizer()
        texts = [
            "文本一：人工智能是计算机科学的一个分支。",
            "文本二：机器学习是人工智能的子领域。",
            "文本三：深度学习是机器学习的一种方法。"
        ]
        results = summarizer.batch_summarize(texts)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, SummaryResult)
    
    def test_batch_summarize_with_error(self):
        """测试批量处理包含错误的情况"""
        summarizer = Summarizer()
        texts = ["有效文本", "", "另一个有效文本"]  # 中间有空文本
        results = summarizer.batch_summarize(texts)
        assert len(results) == 3
        # 第二个应该返回错误结果
        assert "失败" in results[1].summary or results[1].model_name == "error"
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        summarizer = Summarizer()
        info = summarizer.get_model_info()
        assert 'model_name' in info
        assert 'device' in info
        assert 'loaded' in info
```

### 4.4 便捷函数测试

```python
# tests/test_summarizer_utils.py

class TestSummarizerUtils:
    """便捷函数测试类"""
    
    def test_summarize_text(self):
        """测试便捷函数"""
        text = "人工智能是计算机科学的一个分支。"
        summary = summarize_text(text)
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_summarize_text_with_params(self):
        """测试带参数的便捷函数"""
        text = "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。" * 10
        summary = summarize_text(text, max_length=50, min_length=20)
        assert isinstance(summary, str)
    
    def test_summarize_text_invalid_input(self):
        """测试无效输入"""
        with pytest.raises(ValueError):
            summarize_text("")
        with pytest.raises(ValueError):
            summarize_text(None)
    
    def test_get_summarizer(self):
        """测试获取单例"""
        s1 = get_summarizer()
        s2 = get_summarizer()
        assert s1 is s2
        assert isinstance(s1, Summarizer)
```

---

## 5. 性能基准测试

### 5.1 模型加载性能

```python
# tests/benchmark/test_model_loading.py

class TestModelLoadingBenchmark:
    """模型加载性能测试"""
    
    def test_first_load_time(self):
        """测试首次加载时间"""
        import time
        
        mm = ModelManager()
        mm._cleanup_model()  # 确保清理
        
        start = time.time()
        mm.load_model()
        elapsed = time.time() - start
        
        # 首次加载应该在 30 秒内完成
        assert elapsed < 30.0
        print(f"首次加载时间: {elapsed:.2f}s")
    
    def test_cached_load_time(self):
        """测试缓存加载时间"""
        import time
        
        mm = ModelManager()
        mm.load_model()  # 预热
        
        start = time.time()
        mm.load_model()  # 应该立即返回缓存
        elapsed = time.time() - start
        
        # 缓存加载应该在 0.1 秒内
        assert elapsed < 0.1
        print(f"缓存加载时间: {elapsed:.4f}s")
    
    def test_memory_usage(self):
        """测试内存占用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 加载前内存
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        mm = ModelManager()
        mm.load_model()
        
        # 加载后内存
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        mem_increase = mem_after - mem_before
        print(f"内存增加: {mem_increase:.2f} MB")
        
        # 模型应该占用合理内存 (< 2GB)
        assert mem_increase < 2048
```

### 5.2 推理性能

```python
# tests/benchmark/test_inference.py

class TestInferenceBenchmark:
    """推理性能测试"""
    
    def test_short_text_inference_speed(self):
        """测试短文本推理速度"""
        import time
        
        summarizer = Summarizer()
        text = "人工智能是计算机科学的一个分支。" * 10
        
        # 预热
        summarizer.summarize(text)
        
        # 正式测试
        times = []
        for _ in range(10):
            start = time.time()
            summarizer.summarize(text)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"短文本平均推理时间: {avg_time:.3f}s")
        
        # 应该在 2 秒内完成
        assert avg_time < 2.0
    
    def test_long_text_inference_speed(self):
        """测试长文本推理速度"""
        import time
        
        summarizer = Summarizer()
        text = "人工智能是计算机科学的一个分支。" * 1000
        
        start = time.time()
        result = summarizer.summarize(text)
        elapsed = time.time() - start
        
        print(f"长文本推理时间: {elapsed:.3f}s, 分段数: {result.num_chunks}")
        
        # 长文本应该在 10 秒内完成
        assert elapsed < 10.0
    
    def test_throughput(self):
        """测试吞吐量"""
        import time
        
        summarizer = Summarizer()
        texts = ["测试文本 " * 50 for _ in range(20)]
        
        start = time.time()
        results = summarizer.batch_summarize(texts)
        elapsed = time.time() - start
        
        throughput = len(texts) / elapsed
        print(f"吞吐量: {throughput:.2f} 文本/秒")
        
        # 吞吐量应该 > 0.5 文本/秒
        assert throughput > 0.5
```

### 5.3 缓存性能

```python
# tests/benchmark/test_cache.py

class TestCacheBenchmark:
    """缓存性能测试"""
    
    def test_cache_hit_performance(self):
        """测试缓存命中性能"""
        import time
        
        text = "缓存测试文本 " * 50
        
        # 第一次调用（缓存未命中）
        start = time.time()
        summarize_text(text, use_cache=True)
        miss_time = time.time() - start
        
        # 第二次调用（缓存命中）
        start = time.time()
        summarize_text(text, use_cache=True)
        hit_time = time.time() - start
        
        print(f"缓存未命中: {miss_time:.3f}s, 缓存命中: {hit_time:.3f}s")
        print(f"加速比: {miss_time/hit_time:.1f}x")
        
        # 缓存命中应该快 10 倍以上
        assert hit_time < miss_time / 10
```

---

## 6. 功能对比测试

### 6.1 摘要质量对比

```python
# tests/comparison/test_quality.py

class TestQualityComparison:
    """摘要质量对比测试"""
    
    def test_summary_length_control(self):
        """测试摘要长度控制"""
        summarizer = Summarizer()
        
        text = "人工智能是计算机科学的一个分支。" * 100
        
        # 测试不同长度配置
        configs = [
            SummaryConfig(max_length=30, min_length=10),
            SummaryConfig(max_length=100, min_length=50),
            SummaryConfig(max_length=150, min_length=100),
        ]
        
        for config in configs:
            result = summarizer.summarize(text, config)
            word_count = len(result.summary)
            
            # 摘要长度应该在配置范围内
            assert config.min_length <= word_count <= config.max_length * 1.5
    
    def test_compression_ratio(self):
        """测试压缩比合理性"""
        summarizer = Summarizer()
        
        text = "人工智能是计算机科学的一个分支。" * 100
        result = summarizer.summarize(text)
        
        # 压缩比应该在 10%-50% 之间
        assert 0.1 <= result.compression_ratio <= 0.5
    
    def test_content_preservation(self):
        """测试内容保留度"""
        summarizer = Summarizer()
        
        # 包含关键信息的文本
        text = """
        人工智能（AI）是计算机科学的一个分支。机器学习是 AI 的子领域。
        深度学习是机器学习的一种方法。神经网络是深度学习的核心组件。
        """
        
        result = summarizer.summarize(text)
        
        # 摘要应该包含关键概念
        key_concepts = ['人工智能', '机器学习', '深度学习']
        contained = sum(1 for concept in key_concepts if concept in result.summary)
        
        # 至少包含 2 个关键概念
        assert contained >= 2
```

### 6.2 功能完整性测试

```python
# tests/comparison/test_functionality.py

class TestFunctionalityComparison:
    """功能完整性对比测试"""
    
    def test_api_compatibility(self):
        """测试 API 兼容性"""
        # 测试便捷函数
        summary1 = summarize_text("测试文本")
        assert isinstance(summary1, str)
        
        # 测试完整接口
        summarizer = get_summarizer()
        result = summarizer.summarize("测试文本")
        assert isinstance(result, SummaryResult)
        
        # 测试批量接口
        results = summarizer.batch_summarize(["文本1", "文本2"])
        assert len(results) == 2
    
    def test_result_structure(self):
        """测试结果结构完整性"""
        summarizer = Summarizer()
        result = summarizer.summarize("测试文本")
        
        # 验证所有字段
        assert hasattr(result, 'summary')
        assert hasattr(result, 'original_length')
        assert hasattr(result, 'summary_length')
        assert hasattr(result, 'compression_ratio')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'num_chunks')
        assert hasattr(result, 'model_name')
        
        # 验证 to_dict 方法
        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'summary' in d
```

---

## 7. 边界条件测试

### 7.1 输入边界

```python
# tests/boundary/test_input_boundary.py

class TestInputBoundary:
    """输入边界测试"""
    
    def test_minimum_length_text(self):
        """测试最小长度文本"""
        summarizer = Summarizer()
        text = "短"
        result = summarizer.summarize(text)
        assert isinstance(result, SummaryResult)
    
    def test_maximum_length_text(self):
        """测试最大长度文本"""
        summarizer = Summarizer()
        # 生成超长文本 (10万字符)
        text = "测试文本。" * 10000
        result = summarizer.summarize(text)
        assert isinstance(result, SummaryResult)
        assert result.num_chunks > 10
    
    def test_special_characters(self):
        """测试特殊字符"""
        summarizer = Summarizer()
        texts = [
            "包含emoji 😀🎉的文本",
            "包含HTML <div>标签</div>的文本",
            "包含URL https://example.com 的文本",
            "包含代码 `print('hello')` 的文本",
        ]
        for text in texts:
            result = summarizer.summarize(text)
            assert isinstance(result, SummaryResult)
    
    def test_multilingual(self):
        """测试多语言"""
        summarizer = Summarizer()
        texts = [
            "English text for summarization.",
            "日本語のテキスト。",
            "한국어 텍스트.",
            "Deutscher Text.",
        ]
        for text in texts:
            result = summarizer.summarize(text)
            assert isinstance(result, SummaryResult)
```

### 7.2 并发边界

```python
# tests/boundary/test_concurrency.py

class TestConcurrencyBoundary:
    """并发边界测试"""
    
    def test_concurrent_summarize(self):
        """测试并发摘要"""
        import threading
        
        summarizer = Summarizer()
        results = []
        errors = []
        
        def worker(text):
            try:
                result = summarizer.summarize(text)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(f"并发测试文本 {i}。" * 50,))
            threads.append(t)
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 所有线程应该成功
        assert len(results) == 5
        assert len(errors) == 0
    
    def test_stress_test(self):
        """压力测试"""
        summarizer = Summarizer()
        
        # 快速连续调用 50 次
        for i in range(50):
            result = summarizer.summarize(f"压力测试文本 {i}。" * 20)
            assert isinstance(result, SummaryResult)
```

---

## 8. 异常处理测试

```python
# tests/exception/test_exceptions.py

class TestExceptionHandling:
    """异常处理测试"""
    
    def test_model_load_error(self):
        """测试模型加载异常"""
        mm = ModelManager()
        with pytest.raises(ModelLoadError):
            mm.load_model(model_name="invalid_model_name_12345")
    
    def test_text_processing_error(self):
        """测试文本处理异常"""
        with pytest.raises(TextProcessingError):
            TextPreprocessor.clean_text("")
        
        with pytest.raises(TextProcessingError):
            TextPreprocessor.clean_text(None)
    
    def test_inference_error_handling(self):
        """测试推理异常处理"""
        summarizer = Summarizer()
        
        # 批量处理中的单个错误不应影响其他
        texts = ["有效文本", None, "另一个有效文本"]
        results = summarizer.batch_summarize(texts)
        
        assert len(results) == 3
        assert results[0].model_name != "error"
        assert results[1].model_name == "error"
        assert results[2].model_name != "error"
```

---

## 9. 测试执行计划

### 9.1 测试优先级

**P0 (阻塞性)**:
- 单元测试: test_summarize_short_text, test_singleton
- 边界测试: test_minimum_length_text, test_maximum_length_text
- 异常测试: test_text_processing_error

**P1 (重要)**:
- 所有单元测试
- 性能测试: test_short_text_inference_speed
- 功能测试: test_api_compatibility

**P2 (一般)**:
- 详细性能基准测试
- 并发压力测试
- 多语言测试

### 9.2 执行命令

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/test_*.py -v

# 运行性能测试
pytest tests/benchmark/ -v --benchmark-only

# 运行边界测试
pytest tests/boundary/ -v

# 生成覆盖率报告
pytest tests/ --cov=backend.models.summarizer --cov-report=html

# 运行特定测试
pytest tests/test_summarizer.py::TestSummarizer::test_summarize_short_text -v
```

### 9.3 CI/CD 集成

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=backend.models.summarizer --cov-fail-under=80
      
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

---

## 10. 验收标准

### 10.1 功能验收
- [ ] 所有 P0 测试用例通过
- [ ] 代码覆盖率 >= 80%
- [ ] API 完全兼容原有接口

### 10.2 性能验收
- [ ] 短文本推理时间 < 2s
- [ ] 长文本推理时间 < 10s
- [ ] 内存占用 < 2GB
- [ ] 缓存加速比 > 10x

### 10.3 质量验收
- [ ] 摘要压缩比在合理范围 (10%-50%)
- [ ] 关键信息保留率 > 80%
- [ ] 并发调用无错误

---

## 11. 附录

### 11.1 测试数据

```python
# tests/data/sample_texts.py

SAMPLE_SHORT_TEXT = """
人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。
"""

SAMPLE_MEDIUM_TEXT = """
人工智能（Artificial Intelligence，AI）是指由人制造出来的系统所表现出来的智能。
人工智能的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
"""

SAMPLE_LONG_TEXT = """
人工智能是计算机科学的一个分支。""" * 1000
```

### 11.2 性能基线

| 指标 | 基线值 | 目标值 |
|------|--------|--------|
| 模型加载时间 | 30s | < 30s |
| 短文本推理 | 2s | < 2s |
| 长文本推理 | 10s | < 10s |
| 内存占用 | 2GB | < 2GB |
| 代码覆盖率 | 60% | > 80% |
