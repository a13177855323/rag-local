#!/usr/bin/env python
"""
后端功能测试脚本
"""
import sys
sys.path.insert(0, '.')

def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)
    
    try:
        from backend.config import settings
        print("✓ backend.config 导入成功")
    except Exception as e:
        print(f"✗ backend.config 导入失败: {e}")
        return False
    
    try:
        from backend.utils.document_processor import DocumentProcessor
        print("✓ backend.utils.document_processor 导入成功")
    except Exception as e:
        print(f"✗ backend.utils.document_processor 导入失败: {e}")
        return False
    
    try:
        from backend.models.embedding_model import EmbeddingModel
        print("✓ backend.models.embedding_model 导入成功")
    except Exception as e:
        print(f"✗ backend.models.embedding_model 导入失败: {e}")
        return False
    
    try:
        from backend.models.llm_model import LLMModel
        print("✓ backend.models.llm_model 导入成功")
    except Exception as e:
        print(f"✗ backend.models.llm_model 导入失败: {e}")
        return False
    
    try:
        from backend.services.vector_store import VectorStore
        print("✓ backend.services.vector_store 导入成功")
    except Exception as e:
        print(f"✗ backend.services.vector_store 导入失败: {e}")
        return False
    
    try:
        from backend.services.rag_service import RAGService
        print("✓ backend.services.rag_service 导入成功")
    except Exception as e:
        print(f"✗ backend.services.rag_service 导入失败: {e}")
        return False
    
    print("\n所有模块导入成功!")
    return True


def test_config():
    """测试配置"""
    print("\n" + "=" * 60)
    print("测试配置...")
    print("=" * 60)
    
    from backend.config import settings
    
    print(f"APP_NAME: {settings.APP_NAME}")
    print(f"EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
    print(f"LLM_MODEL: {settings.LLM_MODEL}")
    print(f"VECTOR_DB_PATH: {settings.VECTOR_DB_PATH}")
    print(f"UPLOAD_DIR: {settings.UPLOAD_DIR}")
    
    print("\n配置加载成功!")
    return True


def test_vector_store():
    """测试向量存储"""
    print("\n" + "=" * 60)
    print("测试向量存储...")
    print("=" * 60)
    
    from backend.services.vector_store import VectorStore
    
    try:
        vs = VectorStore()
        print(f"✓ VectorStore 实例化成功")
        print(f"  - 文档数量: {vs.get_document_count()}")
        
        # 测试添加和搜索
        import numpy as np
        
        # 创建测试嵌入
        dim = 1024
        test_embeddings = np.random.rand(2, dim).astype(np.float32)
        test_docs = [
            {
                "id": "test_1",
                "content": "这是一个测试文档",
                "metadata": {"filename": "test.txt", "chunk_id": 0}
            },
            {
                "id": "test_2",
                "content": "这是另一个测试文档",
                "metadata": {"filename": "test.txt", "chunk_id": 1}
            }
        ]
        
        vs.add_documents(test_embeddings, test_docs)
        print(f"✓ 添加测试文档成功")
        
        # 测试搜索
        query_embedding = np.random.rand(dim).astype(np.float32)
        results = vs.search(query_embedding, top_k=2)
        print(f"✓ 搜索测试成功，返回 {len(results)} 条结果")
        
        # 清理测试数据
        vs.clear()
        print(f"✓ 清理测试数据成功")
        
    except Exception as e:
        print(f"✗ VectorStore 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_document_processor():
    """测试文档处理器"""
    print("\n" + "=" * 60)
    print("测试文档处理器...")
    print("=" * 60)
    
    from backend.utils.document_processor import DocumentProcessor
    
    try:
        dp = DocumentProcessor()
        print(f"✓ DocumentProcessor 实例化成功")
        
        # 测试文本分块
        test_text = """这是一段测试文本。"
        它包含多行内容。
        每一行都有不同的信息。
        我们需要测试分块功能是否正常工作。"""
        
        chunks = dp.text_splitter.split_text(test_text)
        print(f"✓ 文本分块成功，生成 {len(chunks)} 个块")
        
    except Exception as e:
        print(f"✗ DocumentProcessor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """主测试函数"""
    print("后端功能测试")
    print("=" * 60)
    
    all_passed = True
    
    # 运行所有测试
    all_passed &= test_imports()
    all_passed &= test_config()
    all_passed &= test_document_processor()
    all_passed &= test_vector_store()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败! ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
