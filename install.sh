#!/bin/bash
#
# 本地RAG知识库系统安装脚本
#

echo "========================================"
echo "本地RAG知识库系统安装脚本"
echo "========================================"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3命令，请先安装Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "检测到Python版本: $PYTHON_VERSION"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "错误: 创建虚拟环境失败"
        exit 1
    fi
fi

# 激活虚拟环境
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装Python依赖
echo "安装Python依赖..."
pip install fastapi uvicorn python-multipart torch transformers sentence-transformers faiss-cpu PyPDF2 python-docx markdown langchain pydantic-settings

if [ $? -ne 0 ]; then
    echo "错误: 安装Python依赖失败"
    exit 1
fi

# 安装Node.js依赖
echo "安装Node.js依赖..."
npm install

if [ $? -ne 0 ]; then
    echo "错误: 安装Node.js依赖失败"
    exit 1
fi

# 构建前端
echo "构建前端应用..."
npm run build

if [ $? -ne 0 ]; then
    echo "错误: 构建前端应用失败"
    exit 1
fi

echo ""
echo "========================================"
echo "安装完成!"
echo "========================================"
echo ""
echo "启动系统:"
echo "  1. 启动后端服务: ./start.sh"
echo "  2. 在新终端启动前端: npm run dev"
echo ""
echo "访问地址:"
echo "  - 前端界面: http://localhost:5173"
echo "  - 后端API: http://localhost:8000"
echo "  - API文档: http://localhost:8000/docs"
echo ""
echo "注意: 首次运行时会自动下载模型文件，请确保网络连接正常。"
echo "模型文件较大，请耐心等待下载完成。"
echo ""
echo "支持的文件格式: PDF, DOCX, MD, TXT"
echo "========================================"
