#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEU-DET 钢材表面缺陷检测训练脚本 (支持 Google Colab / 本地 GPU / 服务器)
支持原版 YOLOv8 以及融合 Coordinate Attention / Swin Transformer 的魔改模型
"""

import os
import sys
import argparse
from pathlib import Path

# 将当前项目根目录加入 sys.path，确保优先加载本目录下的定制 ultralytics 模块
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import torch
# 彻底解决 PyTorch 2.6+ 默认 weights_only=True 导致加载 YOLO 权重报 _pickle.UnpicklingError 的问题
_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="NEU-DET YOLOv8 训练脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="模型配置 yaml 或预训练权重 pt 路径 (默认: yolov8n.pt，可选 yolov8s.pt, 或 cfg/models/v8/yolov8__CoorAtt.yaml 等)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(CURRENT_DIR / "data" / "data.yaml"),
        help="数据集配置文件路径 (默认: data/data.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数 (默认: 50，Colab 快速验证可设为 10~50，最终训练可设 200~400)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch 大小 (默认: 16，显存不足可降为 8)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像分辨率 (默认: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练设备 (默认: '0' 即第一块 GPU；若无 GPU 可填 'cpu')",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(CURRENT_DIR / "runs" / "train"),
        help="训练结果与权重保存目录",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="neu_det_exp",
        help="实验名称",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=str(CURRENT_DIR / "yolov8n.pt"),
        help="预训练权重 (当 --model 传入 yaml 时用于权重迁移)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # 自动设备兼容性检测
    device = str(args.device)
    if device != "cpu" and not torch.cuda.is_available():
        print("\n" + "⚠️ " * 20)
        print("【警告】检测到当前运行环境中没有可用的 NVIDIA GPU / CUDA 驱动！")
        print("  - 若您在 Google Colab 运行：请在顶部菜单栏选择【代码执行程序】->【更改运行时类型】-> 硬件加速器选择【T4 GPU】并保存，然后重新连接！")
        print("  - 若您在本地机器运行：已自动回退到 'cpu' 模式继续运行（速度较慢）。")
        print("⚠️ " * 20 + "\n")
        device = "cpu"

    print("=" * 60)
    print("🚀 启动 NEU-DET 缺陷检测训练任务")
    print(f"📁 项目主目录: {CURRENT_DIR}")
    print(f"⚙️ 模型配置: {args.model}")
    print(f"📊 数据配置: {args.data}")
    print(f"🎯 Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}, Device: {device}")
    print("=" * 60)

    # 兼容相对路径或文件名
    model_path = args.model
    if not os.path.exists(model_path):
        candidate_cfg = CURRENT_DIR / "ultralytics" / "cfg" / "models" / "v8" / args.model
        candidate_pt = CURRENT_DIR / args.model
        if candidate_cfg.exists():
            model_path = str(candidate_cfg)
        elif candidate_pt.exists():
            model_path = str(candidate_pt)

    # 加载模型
    if model_path.endswith(".yaml"):
        print(f"正在基于 YAML 构建网络架构: {model_path}")
        model = YOLO(model_path)
        if os.path.exists(args.pretrained):
            print(f"迁移官方预训练权重: {args.pretrained}")
            model.load(args.pretrained)
    else:
        print(f"加载预训练权重模型: {model_path}")
        model = YOLO(model_path)

    # 启动训练
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    print("\n✅ 训练完成！权重已保存在:", os.path.join(args.project, args.name, "weights"))


if __name__ == "__main__":
    main()
