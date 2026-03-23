import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 基础配置
    APP_NAME: str = "Local RAG System"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 模型配置
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    LLM_MODEL: str = "Qwen/Qwen1.5-0.5B-Chat"
    DEVICE: str = "cpu"

    # 向量数据库配置
    VECTOR_DB_PATH: str = "./data/vector_store"
    VECTOR_DIMENSION: int = 1024

    # 文档处理配置
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # RAG配置
    TOP_K: int = 5
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.7

    # 代码检索配置
    CODE_SEARCH_TOP_K: int = 3
    CODE_BOOST_FACTOR: float = 1.5
    ENABLE_CODE_DETECTION: bool = True
    CODE_CONFIDENCE_THRESHOLD: float = 0.3

    # 文件存储配置
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024

    class Config:
        env_file = ".env"

settings = Settings()

# 创建必要的目录
os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
