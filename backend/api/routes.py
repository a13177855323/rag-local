"""
RAG系统API路由模块

提供文档上传、查询、对话历史管理等API接口。
遵循PEP8编码规范，包含完整的异常处理和日志记录。

Author: RAG System
Date: 2026-03-23
"""

import json
import logging
import os
import shutil
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from backend.config import settings
from backend.services.conversation_analyzer import get_conversation_analyzer
from backend.services.rag_service import RAGService


# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG"])
rag_service = RAGService()
conversation_analyzer = get_conversation_analyzer()


class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None


class CreateSessionRequest(BaseModel):
    """创建会话请求模型"""
    title: Optional[str] = None


class DeleteRequest(BaseModel):
    """删除请求模型"""
    filename: str


# ==================== 文档管理接口 ====================

@router.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """
    上传文档文件

    支持的文件类型: .pdf, .docx, .md, .txt, .text
    """
    results = []

    for file in files:
        allowed_extensions = ['.pdf', '.docx', '.md', '.txt', '.text']
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": f"不支持的文件类型。支持的类型: {', '.join(allowed_extensions)}"
            })
            continue

        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            results.append({
                "success": False,
                "filename": file.filename,
                "error": f"保存文件失败: {str(e)}"
            })
            continue

        result = rag_service.ingest_document(file_path)
        results.append(result)

        os.remove(file_path)

    return {"results": results}


@router.post("/query")
async def query(request: QueryRequest):
    """
    查询知识库

    如果没有提供session_id，会自动创建新会话。
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = request.session_id
    if not session_id:
        session_id = rag_service.create_conversation_session()

    result = rag_service.query(
        question=request.question,
        top_k=request.top_k,
        stream=False,
        session_id=session_id
    )
    return result


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    流式查询知识库

    返回Server-Sent Events格式的流式响应。
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = request.session_id
    if not session_id:
        session_id = rag_service.create_conversation_session()

    stream_generator = rag_service.query(
        question=request.question,
        top_k=request.top_k,
        stream=True,
        session_id=session_id
    )

    return StreamingResponse(
        stream_generator,
        media_type="text/event-stream"
    )


@router.get("/documents")
async def get_documents():
    """获取文档列表"""
    documents = rag_service.get_document_list()
    return {"documents": documents}


@router.delete("/documents")
async def delete_document(request: DeleteRequest):
    """删除指定文档"""
    result = rag_service.delete_document(request.filename)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    stats = rag_service.get_stats()
    return stats


@router.delete("/clear")
async def clear_knowledge_base():
    """清空知识库"""
    result = rag_service.clear_knowledge_base()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "message": "RAG系统运行正常"}


# ==================== 对话历史管理接口 ====================

@router.post("/conversations")
async def create_conversation(request: CreateSessionRequest):
    """
    创建对话会话

    如果未提供标题，将自动生成默认标题。
    """
    session_id = rag_service.create_conversation_session(request.title)
    if not session_id:
        raise HTTPException(status_code=400, detail="对话历史功能未启用")

    return {
        "success": True,
        "session_id": session_id,
        "message": "会话创建成功"
    }


@router.get("/conversations")
async def list_conversations():
    """获取所有对话会话列表"""
    sessions = rag_service.get_conversation_sessions()
    return {"sessions": sessions}


@router.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    """
    获取单个会话详情

    包含完整的对话轮次列表。
    """
    session = rag_service.get_conversation_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """删除对话会话"""
    result = rag_service.delete_conversation_session(session_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ==================== 对话导出接口 ====================

@router.get("/conversations/{session_id}/export/markdown")
async def export_conversation_markdown(
    session_id: str,
    include_analysis: bool = Query(True, description="是否包含分析报告")
):
    """
    导出对话为Markdown格式

    返回可下载的Markdown文件。
    """
    content = conversation_analyzer.export_to_markdown(session_id, include_analysis)
    if content.startswith("# 错误"):
        raise HTTPException(status_code=404, detail="会话不存在")

    return PlainTextResponse(
        content=content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.md"
        }
    )


@router.get("/conversations/{session_id}/export/json")
async def export_conversation_json(
    session_id: str,
    include_analysis: bool = Query(True, description="是否包含分析报告")
):
    """
    导出对话为JSON格式

    返回结构化的JSON数据。
    """
    data = conversation_analyzer.export_to_json(session_id, include_analysis)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])

    return JSONResponse(
        content=data,
        headers={
            "Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.json"
        }
    )


@router.get("/conversations/{session_id}/export/csv")
async def export_conversation_csv(session_id: str):
    """
    导出对话为CSV格式

    返回表格化的CSV数据。
    """
    content = conversation_analyzer.export_to_csv(session_id)
    if content.startswith("error"):
        raise HTTPException(status_code=404, detail="会话不存在")

    return PlainTextResponse(
        content=content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.csv"
        }
    )


# ==================== 对话分析接口 ====================

@router.get("/conversations/{session_id}/analysis")
async def analyze_conversation(session_id: str):
    """
    获取对话分析报告

    包含摘要、统计信息、分类分布、质量指标等。
    """
    analysis = conversation_analyzer.analyze_session(session_id)
    if "error" in analysis:
        raise HTTPException(status_code=404, detail=analysis["error"])
    return analysis


@router.get("/conversations/statistics/global")
async def get_global_statistics():
    """
    获取全局统计信息

    包含所有会话的汇总统计数据。
    """
    stats = conversation_analyzer.get_global_statistics()
    return stats
