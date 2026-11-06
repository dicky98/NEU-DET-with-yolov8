#!/usr/bin/env bash
# ==============================================================================
# NEU-DET 工业缺陷检测项目 - Ubuntu 专属环境一键配置脚本 (免 Conda)
# 作用：在当前工程根目录下创建 .env 原生虚拟环境并安装 CUDA PyTorch 及所有依赖
# ==============================================================================

set -e

# 获取脚本所在根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

VENV_DIR="${PROJECT_DIR}/.env"

echo "======================================================================"
echo ">>> [1/4] 检查系统 Python 3 环境..."
echo "======================================================================"

if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 python3，请先在 Ubuntu 系统安装: sudo apt update && sudo apt install -y python3 python3-pip python3-venv"
    exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[系统信息] Python 版本: ${PYTHON_VER}"

# 检查是否安装了 python3-venv 模块
if ! python3 -m venv -h &> /dev/null; then
    echo "[错误] python3-venv 未安装，请执行: sudo apt update && sudo apt install -y python3-venv python3-pip"
    exit 1
fi

echo "======================================================================"
echo ">>> [2/4] 在项目根目录下创建专用 .env 虚拟环境..."
echo "======================================================================"

if [ -d "${VENV_DIR}" ]; then
    echo "[提示] 检测到已存在 ${VENV_DIR}，保留并复用现有虚拟环境。"
else
    echo "[创建] 正在创建虚拟环境: ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
    echo "[完成] 虚拟环境创建成功。"
fi

# 激活虚拟环境
source "${VENV_DIR}/bin/activate"

# 升级 pip 工具链
echo "[升级] 升级 pip, setuptools, wheel ..."
pip install --upgrade pip setuptools wheel -q

echo "======================================================================"
echo ">>> [3/4] 安装 GPU 版本 PyTorch (CUDA 11.8 / 12.x 兼容)..."
echo "======================================================================"

# 默认安装广泛稳定的 CUDA 11.8 版 PyTorch (若服务器为更高版本显卡驱动均向下兼容)
# 如需指定 CUDA 12.1 可修改为 cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "======================================================================"
echo ">>> [4/4] 安装项目依赖与 ONNX 导出工具..."
echo "======================================================================"

pip install -r "${PROJECT_DIR}/requirements.txt"
pip install onnx onnxsim

echo ""
echo "======================================================================"
echo ">>> [自检] 验证 PyTorch 与 GPU 设备识别状态..."
echo "======================================================================"

python3 -c "import torch; print(f'CUDA 可用性 : {torch.cuda.is_available()}'); print(f'GPU 设备名 : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无 (仅 CPU)\"}'); print(f'PyTorch版本 : {torch.__version__}')"

echo ""
echo "======================================================================"
echo ">>> 环境准备就绪！后续在 Ubuntu 启动训练的常用方式："
echo "    1. 手动运行: source .env/bin/activate && python3 train_industrial.py"
echo "    2. 快速启动: ./run_train.sh"
echo "======================================================================"
