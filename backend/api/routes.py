from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import json
from backend.config import settings
from backend.services.rag_service import RAGService
from backend.services.conversation_analyzer import get_conversation_analyzer

router = APIRouter(prefix="/api", tags=["RAG"])
rag_service = RAGService()
conversation_analyzer = get_conversation_analyzer()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None  # 对话会话ID

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None

class DeleteRequest(BaseModel):
    filename: str

@router.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """上传文档文件"""
    results = []
    
    for file in files:
        # 检查文件类型
        allowed_extensions = ['.pdf', '.docx', '.md', '.txt', '.text']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": f"不支持的文件类型。支持的类型: {', '.join(allowed_extensions)}"
            })
            continue
        
        # 保存文件
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": f"保存文件失败: {str(e)}"
            })
            continue
        
        # 处理文档
        result = rag_service.ingest_document(file_path)
        results.append(result)
        
        # 清理临时文件
        os.remove(file_path)
    
    return {"results": results}

@router.post("/query")
async def query(request: QueryRequest):
    """查询知识库"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 如果没有session_id，创建新会话
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
    """流式查询知识库"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 如果没有session_id，创建新会话
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


# ========== 对话历史管理接口 ==========

@router.post("/conversations")
async def create_conversation(request: CreateSessionRequest):
    """创建对话会话"""
    session_id = rag_service.create_conversation_session(request.title)
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
    """获取单个会话详情"""
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


# ========== 对话导出接口 ==========

@router.get("/conversations/{session_id}/export/markdown")
async def export_conversation_markdown(
    session_id: str,
    include_analysis: bool = Query(True, description="是否包含分析报告")
):
    """导出对话为 Markdown 格式"""
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
    """导出对话为 JSON 格式"""
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
    """导出对话为 CSV 格式"""
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


# ========== 对话分析接口 ==========

@router.get("/conversations/{session_id}/analysis")
async def analyze_conversation(session_id: str):
    """获取对话分析报告"""
    analysis = conversation_analyzer.analyze_session(session_id)
    if "error" in analysis:
        raise HTTPException(status_code=404, detail=analysis["error"])
    return analysis

@router.get("/conversations/statistics/global")
async def get_global_statistics():
    """获取全局统计信息"""
    stats = conversation_analyzer.get_global_statistics()
    return stats
