from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from backend.config import settings
from backend.api.routes import router

app = FastAPI(
    title=settings.APP_NAME,
    description="本地离线RAG知识库系统API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)

@app.get("/")
async def root():
    return {
        "message": "欢迎使用本地离线RAG知识库系统",
        "docs": "/docs",
        "health": "/api/health"
    }

def main():
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1  # 单worker避免多进程模型加载问题
    )

if __name__ == "__main__":
    main()
