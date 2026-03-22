#!/bin/bash
#
# 本地RAG知识库系统启动脚本
#

echo "========================================"
echo "本地RAG知识库系统启动脚本"
echo "========================================"

# 检查Python虚拟环境
if [ ! -d "venv" ]; then
    echo "错误: 虚拟环境不存在，请先运行安装脚本"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 启动后端服务
echo "启动后端服务..."
echo "后端服务将在 http://localhost:8000 启动"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "启动前端服务请在新终端运行: npm run dev"
echo "前端服务将在 http://localhost:5173 启动"
echo ""
echo "按 Ctrl+C 停止服务"
echo "========================================"

python -m backend.main
