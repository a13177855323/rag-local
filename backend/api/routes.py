from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from backend.config import settings
from backend.services.rag_service import RAGService

router = APIRouter(prefix="/api", tags=["RAG"])
rag_service = RAGService()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    enable_code_rag: Optional[bool] = True

class CodeDetectRequest(BaseModel):
    content: str

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
    
    result = rag_service.query(
        question=request.question,
        top_k=request.top_k,
        stream=False,
        enable_code_rag=request.enable_code_rag
    )
    return result

@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """流式查询知识库"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    stream_generator = rag_service.query(
        question=request.question,
        top_k=request.top_k,
        stream=True,
        enable_code_rag=request.enable_code_rag
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

@router.post("/code/query")
async def code_query(request: QueryRequest):
    """代码专属查询接口 - 专门用于代码相关问题"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    result = rag_service.code_query(
        question=request.question,
        top_k=request.top_k,
        stream=False
    )
    return result

@router.post("/code/query/stream")
async def code_query_stream(request: QueryRequest):
    """代码专属流式查询接口"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    stream_generator = rag_service.code_query(
        question=request.question,
        top_k=request.top_k,
        stream=True
    )
    
    return StreamingResponse(
        stream_generator,
        media_type="text/event-stream"
    )

@router.post("/code/detect")
async def detect_code(request: CodeDetectRequest):
    """检测文本中是否包含Python代码"""
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="内容不能为空")
    
    result = rag_service.detect_code_in_document(request.content)
    return result

@router.get("/code/check-question")
async def check_code_question(question: str):
    """检测问题是否与代码相关"""
    if not question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    is_code = rag_service.code_rag_handler.is_code_question(question)
    return {
        "is_code_question": is_code,
        "question": question
    }
