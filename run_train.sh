#!/usr/bin/env bash
# ==============================================================================
# NEU-DET 工业缺陷检测项目 - 快速启动训练脚本
# 作用：自动激活当前项目的 .env 虚拟环境并执行工业级训练与一键导出
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

VENV_DIR="${PROJECT_DIR}/.env"

# 检查 .env 是否存在
if [ ! -d "${VENV_DIR}" ]; then
    echo "[错误] 未检测到 .env 虚拟环境目录！请先执行: bash setup_env.sh 进行初始化。"
    exit 1
fi

# 激活 .env 虚拟环境
source "${VENV_DIR}/bin/activate"

# 默认参数（支持从命令行直接传入覆盖）
EPOCHS=${1:-150}
BATCH=${2:-16}
IMGSZ=${3:-640}
DEVICE=${4:-0}

echo "======================================================================"
echo ">>> 启动训练环境: ${VENV_DIR}"
echo ">>> 训练轮次(Epochs): ${EPOCHS} | 批次(Batch): ${BATCH} | 分辨率: ${IMGSZ} | GPU设备: ${DEVICE}"
echo "======================================================================"

python3 train_industrial.py \
    --weights yolov8s.pt \
    --data data/data.yaml \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --imgsz "${IMGSZ}" \
    --device "${DEVICE}"
