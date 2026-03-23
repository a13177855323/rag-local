"""
RAG系统API路由模块

提供RESTful API接口，包括：
- 文档上传与管理
- RAG查询接口（支持流式响应）
- 对话会话管理
- 对话导出与分析
- 系统状态与统计

实现了统一的错误处理、输入验证和响应格式规范。
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import os
import shutil
import json
import uuid
from datetime import datetime
from enum import Enum

from backend.config import settings
from backend.services.rag_service import RAGService
from backend.services.conversation_analyzer import get_conversation_analyzer, SummaryStrategy, SummaryLength
from backend.services.conversation_store import get_conversation_store


# ==================== 枚举类型定义 ====================
class ExportFormat(str, Enum):
    """导出格式枚举"""
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"


class APIRoutes(str, Enum):
    """API路由常量"""
    UPLOAD = "/upload"
    QUERY = "/query"
    QUERY_STREAM = "/query/stream"
    DOCUMENTS = "/documents"
    STATS = "/stats"
    CLEAR = "/clear"
    HEALTH = "/health"
    CONVERSATIONS = "/conversations"
    CONVERSATION_ANALYSIS = "/conversations/{session_id}/analysis"
    CONVERSATION_EXPORT = "/conversations/{session_id}/export/{format}"
    GLOBAL_STATS = "/conversations/statistics/global"


# ==================== 请求模型 ====================
class BaseRequest(BaseModel):
    """基础请求模型，提供通用验证方法"""
    class Config:
        extra = "forbid"  # 禁止额外字段


class QueryRequest(BaseRequest):
    """查询请求模型"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="返回相关文档数量 (1-20)")
    stream: Optional[bool] = Field(False, description="是否使用流式响应")
    session_id: Optional[str] = Field(None, description="对话会话ID，用于多轮对话")

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        """验证问题内容"""
        v = v.strip()
        if not v:
            raise ValueError("问题不能为空")
        if len(v) > 2000:
            raise ValueError("问题长度不能超过2000字符")
        return v


class CreateSessionRequest(BaseRequest):
    """创建会话请求模型"""
    title: Optional[str] = Field(None, max_length=100, description="会话标题")

    @field_validator('title')
    @classmethod
    def validate_title(cls, v: Optional[str]) -> Optional[str]:
        """验证标题"""
        if v is not None:
            v = v.strip()
            if len(v) > 100:
                raise ValueError("标题长度不能超过100字符")
        return v


class DeleteDocumentRequest(BaseRequest):
    """删除文档请求模型"""
    filename: str = Field(..., min_length=1, description="要删除的文件名")

    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """验证文件名"""
        v = v.strip()
        if not v:
            raise ValueError("文件名不能为空")
        # 防止路径遍历攻击
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError("无效的文件名")
        return v


class SummaryRequest(BaseRequest):
    """摘要生成请求模型"""
    text: str = Field(..., min_length=10, max_length=50000, description="待摘要文本")
    strategy: Optional[SummaryStrategy] = Field(SummaryStrategy.HYBRID, description="摘要策略")
    length: Optional[SummaryLength] = Field(SummaryLength.MEDIUM, description="摘要长度")
    max_length: Optional[int] = Field(None, ge=50, le=5000, description="自定义最大长度")

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """验证文本内容"""
        v = v.strip()
        if len(v) < 10:
            raise ValueError("文本内容至少需要10字符")
        if len(v) > 50000:
            raise ValueError("文本内容不能超过50000字符")
        return v


# ==================== 响应模型 ====================
class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field("", description="响应消息")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DataResponse(BaseResponse):
    """带数据的响应模型"""
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = Field(False, const=False)
    error_code: str = Field(..., description="错误码")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")


# ==================== 错误处理中间件 ====================
class APIErrorHandler:
    """统一API错误处理器"""

    # 错误码定义
    ERROR_CODES = {
        "VALIDATION_ERROR": "VALIDATION_ERROR",
        "FILE_TYPE_ERROR": "FILE_TYPE_ERROR",
        "FILE_SIZE_ERROR": "FILE_SIZE_ERROR",
        "PROCESSING_ERROR": "PROCESSING_ERROR",
        "NOT_FOUND": "NOT_FOUND",
        "INTERNAL_ERROR": "INTERNAL_ERROR",
        "EMPTY_CONTENT": "EMPTY_CONTENT",
        "CONFLICT": "CONFLICT"
    }

    @staticmethod
    def create_error_response(
        error_code: str,
        message: str,
        status_code: int = 400,
        error_details: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """
        创建标准化的错误响应

        Args:
            error_code: 错误码
            message: 错误消息
            status_code: HTTP状态码
            error_details: 错误详情

        Returns:
            HTTPException: 标准化的HTTP异常
        """
        error_response = ErrorResponse(
            message=message,
            error_code=error_code,
            error_details=error_details
        )
        return HTTPException(
            status_code=status_code,
            detail=error_response.dict()
        )

    @staticmethod
    def validation_error(message: str, details: Optional[Dict[str, Any]] = None) -> HTTPException:
        """创建验证错误响应"""
        return APIErrorHandler.create_error_response(
            error_code=APIErrorHandler.ERROR_CODES["VALIDATION_ERROR"],
            message=message,
            status_code=400,
            error_details=details
        )

    @staticmethod
    def file_type_error(message: str) -> HTTPException:
        """创建文件类型错误响应"""
        return APIErrorHandler.create_error_response(
            error_code=APIErrorHandler.ERROR_CODES["FILE_TYPE_ERROR"],
            message=message,
            status_code=400
        )

    @staticmethod
    def not_found(message: str) -> HTTPException:
        """创建资源未找到错误响应"""
        return APIErrorHandler.create_error_response(
            error_code=APIErrorHandler.ERROR_CODES["NOT_FOUND"],
            message=message,
            status_code=404
        )

    @staticmethod
    def processing_error(message: str, details: Optional[Dict[str, Any]] = None) -> HTTPException:
        """创建处理错误响应"""
        return APIErrorHandler.create_error_response(
            error_code=APIErrorHandler.ERROR_CODES["PROCESSING_ERROR"],
            message=message,
            status_code=500,
            error_details=details
        )


# ==================== 服务依赖 ====================
def get_rag_service() -> RAGService:
    """获取RAG服务实例（依赖注入）"""
    return RAGService()


def get_analyzer():
    """获取对话分析器实例"""
    return get_conversation_analyzer()


def get_conversation_store_dep():
    """获取对话存储实例"""
    return get_conversation_store()


# ==================== 路由初始化 ====================
router = APIRouter(prefix="/api", tags=["RAG System"])

# 支持的文件类型
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.md', '.txt', '.text'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


# ==================== 工具函数 ====================
def validate_file(file: UploadFile) -> Optional[str]:
    """
    验证上传文件的合法性

    Args:
        file: 上传的文件

    Returns:
        Optional[str]: 错误消息，验证通过返回None
    """
    # 检查文件类型
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return f"不支持的文件类型 '{file_ext}'。支持的类型: {', '.join(ALLOWED_EXTENSIONS)}"

    # 检查文件大小（如果有content-length头）
    if file.size and file.size > MAX_FILE_SIZE:
        return f"文件过大。最大支持 {MAX_FILE_SIZE // 1024 // 1024}MB"

    return None


# ==================== 文档管理接口 ====================
@router.post(
    APIRoutes.UPLOAD,
    response_model=DataResponse,
    summary="上传文档",
    description="上传一个或多个文档文件到知识库"
)
async def upload_file(
    files: List[UploadFile] = File(..., description="要上传的文件列表"),
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    上传文档文件并添加到知识库

    Args:
        files: 上传的文件列表
        rag_service: RAG服务实例（依赖注入）

    Returns:
        Dict: 上传结果
    """
    if not files:
        raise APIErrorHandler.validation_error("请选择要上传的文件")

    results = []
    success_count = 0
    error_count = 0

    for file in files:
        # 验证文件
        error_msg = validate_file(file)
        if error_msg:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": error_msg
            })
            error_count += 1
            continue

        # 保存并处理文件
        file_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
        try:
            # 保存文件
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 检查实际文件大小
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                raise ValueError(f"文件大小超过限制 ({MAX_FILE_SIZE // 1024 // 1024}MB)")

            # 处理文档
            result = rag_service.ingest_document(file_path)
            results.append(result)

            if result.get("success", False):
                success_count += 1
            else:
                error_count += 1

        except Exception as e:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": f"处理失败: {str(e)}"
            })
            error_count += 1
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)

    # 构建响应消息
    if success_count == 0 and error_count > 0:
        message = f"所有文件处理失败 ({error_count}个)"
    elif error_count == 0:
        message = f"所有文件处理成功 ({success_count}个)"
    else:
        message = f"部分文件处理成功 ({success_count}个成功, {error_count}个失败)"

    return {
        "success": success_count > 0,
        "message": message,
        "data": {
            "results": results,
            "success_count": success_count,
            "error_count": error_count
        }
    }


@router.get(
    APIRoutes.DOCUMENTS,
    response_model=DataResponse,
    summary="获取文档列表",
    description="获取知识库中的所有文档列表"
)
async def get_documents(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """获取文档列表"""
    try:
        documents = rag_service.get_document_list()
        return {
            "success": True,
            "message": f"共找到 {len(documents)} 个文档",
            "data": {"documents": documents}
        }
    except Exception as e:
        raise APIErrorHandler.processing_error(f"获取文档列表失败: {str(e)}")


@router.delete(
    APIRoutes.DOCUMENTS,
    response_model=BaseResponse,
    summary="删除文档",
    description="从知识库中删除指定文档"
)
async def delete_document(
    request: DeleteDocumentRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """删除指定文档"""
    try:
        result = rag_service.delete_document(request.filename)
        if not result["success"]:
            raise APIErrorHandler.not_found(result.get("error", "文档不存在"))

        return {
            "success": True,
            "message": f"文档 '{request.filename}' 删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise APIErrorHandler.processing_error(f"删除文档失败: {str(e)}")


# ==================== 查询接口 ====================
@router.post(
    APIRoutes.QUERY,
    summary="查询知识库",
    description="向知识库提问并获取回答（非流式）"
)
async def query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """查询知识库（非流式响应）"""
    try:
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

        # 添加会话ID到响应
        if isinstance(result, dict):
            result["session_id"] = session_id

        return result

    except Exception as e:
        raise APIErrorHandler.processing_error(f"查询处理失败: {str(e)}")


@router.post(
    APIRoutes.QUERY_STREAM,
    summary="流式查询知识库",
    description="向知识库提问并获取流式回答（用于SSE）"
)
async def query_stream(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> StreamingResponse:
    """查询知识库（流式响应）"""
    try:
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

        # 返回流式响应，包含会话ID信息
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers={
                "X-Session-ID": session_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        raise APIErrorHandler.processing_error(f"查询处理失败: {str(e)}")


# ==================== 对话管理接口 ====================
@router.post(
    APIRoutes.CONVERSATIONS,
    response_model=DataResponse,
    summary="创建对话会话",
    description="创建新的对话会话"
)
async def create_conversation(
    request: CreateSessionRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """创建对话会话"""
    try:
        session_id = rag_service.create_conversation_session(request.title)
        return {
            "success": True,
            "message": "会话创建成功",
            "data": {"session_id": session_id, "title": request.title}
        }
    except Exception as e:
        raise APIErrorHandler.processing_error(f"创建会话失败: {str(e)}")


@router.get(
    APIRoutes.CONVERSATIONS,
    response_model=DataResponse,
    summary="获取会话列表",
    description="获取所有对话会话列表"
)
async def list_conversations(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """获取所有对话会话列表"""
    try:
        sessions = rag_service.get_conversation_sessions()
        return {
            "success": True,
            "message": f"共找到 {len(sessions)} 个会话",
            "data": {"sessions": sessions}
        }
    except Exception as e:
        raise APIErrorHandler.processing_error(f"获取会话列表失败: {str(e)}")


@router.get(
    "/conversations/{session_id}",
    summary="获取会话详情",
    description="获取单个会话的详细信息"
)
async def get_conversation(
    session_id: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """获取单个会话详情"""
    try:
        session = rag_service.get_conversation_session(session_id)
        if not session:
            raise APIErrorHandler.not_found("会话不存在")

        return {
            "success": True,
            "message": "获取会话详情成功",
            "data": {"session": session}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise APIErrorHandler.processing_error(f"获取会话详情失败: {str(e)}")


@router.delete(
    "/conversations/{session_id}",
    response_model=BaseResponse,
    summary="删除会话",
    description="删除指定的对话会话"
)
async def delete_conversation(
    session_id: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """删除对话会话"""
    try:
        result = rag_service.delete_conversation_session(session_id)
        if not result["success"]:
            raise APIErrorHandler.not_found(result.get("error", "会话不存在"))

        return {
            "success": True,
            "message": "会话删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise APIErrorHandler.processing_error(f"删除会话失败: {str(e)}")


# ==================== 摘要功能接口 ====================
@router.post(
    "/summarize",
    response_model=DataResponse,
    summary="生成文档摘要",
    description="对给定文本生成摘要，支持多种策略"
)
async def summarize_text(
    request: SummaryRequest,
    analyzer=Depends(get_analyzer)
) -> Dict[str, Any]:
    """生成文本摘要"""
    try:
        result = analyzer.summarize_document(
            text=request.text,
            strategy=request.strategy,
            length=request.length,
            max_length=request.max_length
        )

        return {
            "success": True,
            "message": "摘要生成成功",
            "data": {
                "summary": result.summary,
                "original_length": result.original_length,
                "summary_length": result.summary_length,
                "compression_ratio": result.compression_ratio,
                "key_points": result.key_points,
                "keywords": result.keywords,
                "strategy": result.strategy,
                "topic": result.topic
            }
        }
    except Exception as e:
        raise APIErrorHandler.processing_error(f"摘要生成失败: {str(e)}")


# ==================== 对话分析接口 ====================
@router.get(
    APIRoutes.CONVERSATION_ANALYSIS,
    summary="对话分析",
    description="获取对话会话的深度分析报告"
)
async def analyze_conversation(
    session_id: str,
    conversation_store=Depends(get_conversation_store_dep),
    analyzer=Depends(get_analyzer)
) -> Dict[str, Any]:
    """获取对话分析报告"""
    try:
        session = conversation_store.get_session(session_id)
        if not session:
            raise APIErrorHandler.not_found("会话不存在")

        analysis = analyzer.get_conversation_insights(session)

        return {
            "success": True,
            "message": "分析完成",
            "data": analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        raise APIErrorHandler.processing_error(f"对话分析失败: {str(e)}")


@router.get(
    APIRoutes.GLOBAL_STATS,
    summary="全局统计",
    description="获取所有对话的全局统计信息"
)
async def get_global_statistics(
    conversation_store=Depends(get_conversation_store_dep)
) -> Dict[str, Any]:
    """获取全局统计信息"""
    try:
        stats = conversation_store.get_global_statistics()
        return {
            "success": True,
            "message": "获取统计信息成功",
            "data": stats
        }
    except Exception as e:
        raise APIErrorHandler.processing_error(f"获取统计信息失败: {str(e)}")


# ==================== 对话导出接口 ====================
@router.get(
    APIRoutes.CONVERSATION_EXPORT,
    summary="导出对话",
    description="将会话导出为指定格式（markdown/json/csv）"
)
async def export_conversation(
    session_id: str,
    format: ExportFormat,
    include_analysis: bool = Query(True, description="是否包含分析报告"),
    conversation_store=Depends(get_conversation_store_dep),
    analyzer=Depends(get_analyzer)
):
    """导出对话为指定格式"""
    try:
        session = conversation_store.get_session(session_id)
        if not session:
            raise APIErrorHandler.not_found("会话不存在")

        if format == ExportFormat.MARKDOWN:
            content = conversation_store.export_session(session_id, format_type='dict')
            # 生成Markdown内容
            md_content = [f"# {session.title}\n"]
            for i, turn in enumerate(session.turns, 1):
                md_content.append(f"## 第{i}轮对话")
                md_content.append(f"**用户问：** {turn.question}\n")
                md_content.append(f"**助手答：** {turn.answer}\n")
                if turn.sources:
                    md_content.append("**来源参考：**")
                    for src in turn.sources:
                        md_content.append(f"- [{src.filename}]: {src.content[:100]}...")
                    md_content.append("")

            content_str = "\n".join(md_content)
            return PlainTextResponse(
                content=content_str,
                media_type="text/markdown",
                headers={
                    "Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.md"
                }
            )

        elif format == ExportFormat.JSON:
            data = conversation_store.export_session(session_id, format_type='dict')
            return JSONResponse(
                content=data,
                headers={
                    "Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.json"
                }
            )

        elif format == ExportFormat.CSV:
            # 生成CSV内容
            csv_lines = ["轮次,问题,回答,响应时间(ms),分类,质量分数"]
            for i, turn in enumerate(session.turns, 1):
                # 处理CSV转义
                question = turn.question.replace('"', '""').replace('\n', ' ')
                answer = turn.answer.replace('"', '""').replace('\n', ' ')
                csv_lines.append(
                    f'{i},"{question}","{answer}",{turn.response_time_ms},{turn.category.value},{turn.quality_score}')

            content_str = "\n".join(csv_lines)
            return PlainTextResponse(
                content=content_str,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.csv"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        raise APIErrorHandler.processing_error(f"导出失败: {str(e)}")


# ==================== 系统状态接口 ====================
@router.get(
    APIRoutes.STATS,
    response_model=DataResponse,
    summary="系统统计",
    description="获取RAG系统的统计信息"
)
async def get_stats(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """获取系统统计信息"""
    try:
        stats = rag_service.get_stats()
        return {
            "success": True,
            "message": "获取统计信息成功",
            "data": stats
        }
    except Exception as e:
        raise APIErrorHandler.processing_error(f"获取统计信息失败: {str(e)}")


@router.delete(
    APIRoutes.CLEAR,
    response_model=BaseResponse,
    summary="清空知识库",
    description="清空知识库中的所有文档（危险操作！）"
)
async def clear_knowledge_base(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """清空知识库"""
    try:
        result = rag_service.clear_knowledge_base()
        if not result["success"]:
            raise APIErrorHandler.processing_error(result.get("error", "清空知识库失败"))

        return {
            "success": True,
            "message": "知识库已清空"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise APIErrorHandler.processing_error(f"清空知识库失败: {str(e)}")


@router.get(
    APIRoutes.HEALTH,
    response_model=BaseResponse,
    summary="健康检查",
    description="检查RAG系统运行状态"
)
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    return {
        "success": True,
        "message": "RAG系统运行正常"
    }


@router.get(
    "/info",
    response_model=DataResponse,
    summary="系统信息",
    description="获取RAG系统的基本信息"
)
async def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    return {
        "success": True,
        "message": "获取系统信息成功",
        "data": {
            "system_name": "Local RAG System",
            "version": "1.0.0",
            "supported_formats": list(ALLOWED_EXTENSIONS),
            "max_file_size_mb": MAX_FILE_SIZE // 1024 // 1024,
            "summary_strategies": [s.value for s in SummaryStrategy],
            "export_formats": [f.value for f in ExportFormat],
            "api_prefix": "/api"
        }
    }
