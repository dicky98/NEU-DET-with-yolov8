"""
NEU-DET 工业缺陷检测工程化训练与自动导出一体化脚本
适用于本地调试及 Ubuntu GPU 服务器生产训练。

功能特性：
1. 自动检测与适配 GPU/CPU 硬件环境；
2. 训前数据集完整性防呆校验（样本统计、标注存在性）；
3. 工业场景调优的训练超参数（自适应学习率、Mosaic 后期关闭策略）；
4. 训练完成后自动调用验证集评估、提取最优权重并导出为 ONNX；
5. 生成符合工业部署规范的 model_meta.json 元数据描述文件。
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="NEU-DET YOLOv8 缺陷检测端到端训练与导出")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="预训练模型权重路径（yolov8n.pt / yolov8s.pt / yolov8m.pt）")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=150, help="训练总轮次")
    parser.add_argument("--batch", type=int, default=16, help="单批次大小（显存不足可调至 8 或使用 -1 开启自动批次）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸（640 / 384）")
    parser.add_argument("--device", type=str, default="", help="计算设备（如 0, 0,1 或 cpu；留空自动推断）")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader 线程数")
    parser.add_argument("--project", type=str, default="runs/train_neu_det", help="训练输出保存根目录")
    parser.add_argument("--name", type=str, default="yolov8s_baseline", help="实验项目名称")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="训练结束后是否自动导出 ONNX 模型")
    return parser.parse_args()


def pre_flight_check(project_root: Path, data_yaml_path: Path):
    """
    训前防呆自检：校验数据集路径、图像及标注文件数量
    """
    print("\n" + "=" * 60)
    print(">>> 正在执行训前数据集防呆检查 (Pre-flight Inspection)...")
    print("=" * 60)
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"[错误] 数据集配置文件未找到: {data_yaml_path}")
    
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    
    base_dir = data_yaml_path.parent
    dataset_path = Path(data_cfg.get("path", "."))
    if not dataset_path.is_absolute():
        dataset_path = (project_root / dataset_path).resolve()
        
    train_images_dir = dataset_path / data_cfg["train"]
    val_images_dir = dataset_path / data_cfg["val"]
    
    train_labels_dir = train_images_dir.parent / "labels"
    val_labels_dir = val_images_dir.parent / "labels"
    
    print(f"[检查项 1] 数据集基准路径: {dataset_path}")
    print(f"[检查项 2] 训练集图像目录: {train_images_dir} (存在: {train_images_dir.exists()})")
    print(f"[检查项 3] 验证集图像目录: {val_images_dir} (存在: {val_images_dir.exists()})")
    
    train_imgs = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png")) + list(train_images_dir.glob("*.bmp"))
    val_imgs = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")) + list(val_images_dir.glob("*.bmp"))
    train_labels = list(train_labels_dir.glob("*.txt"))
    val_labels = list(val_labels_dir.glob("*.txt"))
    
    print(f"[统计] 训练集: {len(train_imgs)} 张图片, {len(train_labels)} 个标注标签")
    print(f"[统计] 验证集: {len(val_imgs)} 张图片, {len(val_labels)} 个标注标签")
    print(f"[统计] 缺陷类别列表 ({data_cfg.get('nc')} 类): {data_cfg.get('names')}")
    
    if len(train_imgs) == 0 or len(val_imgs) == 0:
        raise RuntimeError("[错误] 图像目录为空，请确认数据是否完整解压并拷贝到位！")
    if len(train_labels) == 0:
        raise RuntimeError("[错误] 训练集标注文件为空，请先运行 xml_to_txt.py 转换标签！")
        
    print(">>> 训前检查全部通过，准备开始训练！\n")


def get_optimal_device(requested_device: str) -> str:
    """
    智能选择与校验计算设备
    """
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[设备推断] 检测到 {gpu_count} 块 NVIDIA GPU，使用 GPU 0 ({gpu_name}) 进行训练")
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[设备推断] 检测到 Apple Silicon GPU (MPS)")
        return "mps"
    else:
        print("[设备推断] 未检测到 GPU，降级至 CPU 运行")
        return "cpu"


def train_pipeline(args):
    project_root = Path(__file__).resolve().parent
    data_yaml_path = (project_root / args.data).resolve()
    
    # 1. 训前防呆自检
    pre_flight_check(project_root, data_yaml_path)
    
    # 2. 设备自适应
    device = get_optimal_device(args.device)
    
    # 3. 初始化模型
    weights_path = (project_root / args.weights).resolve()
    if not weights_path.exists():
        print(f"[提示] 未在本地找到预训练权重 {weights_path}，将由 Ultralytics 自动下载官方权重...")
        model = YOLO(args.weights)
    else:
        print(f"[初始化] 加载预训练模型权重: {weights_path}")
        model = YOLO(str(weights_path))
    
    # 4. 执行工业级超参训练
    print("\n" + "=" * 60)
    print(f">>> 开始启动 YOLOv8 工业缺陷检测训练流程")
    print(f">>> 超参配置: Epochs={args.epochs}, Batch={args.batch}, ImgSz={args.imgsz}, Device={device}")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=str(project_root / args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        close_mosaic=10,        # 最后 10 轮关闭 Mosaic 增强，稳定缺陷边界框定位
        patience=30,            # 30 轮指标不提升触发早停
        save=True,
        save_period=10,         # 每 10 轮保存一次 checkpoint
        val=True,
        plots=True,
        verbose=True
    )
    
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\n>>> 训练完成！耗时: {elapsed_minutes:.2f} 分钟")
    
    # 5. 寻找最优权重并进行独立严谨验证
    save_dir = Path(model.trainer.save_dir)
    best_pt_path = save_dir / "weights" / "best.pt"
    print(f"[最优模型保存路径]: {best_pt_path}")
    
    if best_pt_path.exists():
        print("\n" + "=" * 60)
        print(">>> 正在在独立验证集上评估最优模型指标 (Best Checkpoint Validation)...")
        print("=" * 60)
        val_model = YOLO(str(best_pt_path))
        val_metrics = val_model.val(data=str(data_yaml_path), split="val", imgsz=args.imgsz)
        
        map50 = float(val_metrics.box.map50)
        map50_95 = float(val_metrics.box.map)
        print(f"\n[验证集指标总结]")
        print(f" -> mAP@0.50     : {map50 * 100:.2f}%")
        print(f" -> mAP@0.50:0.95: {map50_95 * 100:.2f}%")
        
        # 6. 一键导出 ONNX 并输出工业元数据
        if args.export_onnx:
            print("\n" + "=" * 60)
            print(">>> 正在导出工业部署级 ONNX 动态模型...")
            print("=" * 60)
            onnx_path = val_model.export(
                format="onnx",
                imgsz=args.imgsz,
                dynamic=True,      # 开启动态 Batch 支持
                simplify=True,     # 简化计算图
                opset=12           # 保证 TensorRT / OpenVINO 广泛兼容性
            )
            print(f"[ONNX 导出成功]: {onnx_path}")
            
            # 生成 model_meta.json
            with open(data_yaml_path, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
                
            meta_info = {
                "task": "defect_detection",
                "model_architecture": "YOLOv8s",
                "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_resolution": [args.imgsz, args.imgsz],
                "num_classes": data_cfg.get("nc", 6),
                "class_names": data_cfg.get("names", []),
                "metrics": {
                    "mAP_50": round(map50, 4),
                    "mAP_50_95": round(map50_95, 4)
                },
                "recommended_runtime_thresholds": {
                    "conf_threshold": 0.35,
                    "iou_threshold": 0.45
                },
                "export_files": {
                    "best_torch_checkpoint": str(best_pt_path.name),
                    "onnx_model": Path(onnx_path).name if onnx_path else ""
                }
            }
            meta_path = save_dir / "model_meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_info, f, indent=4, ensure_ascii=False)
            print(f"[元数据文件生成]: {meta_path}")

    print("\n" + "=" * 60)
    print(f">>> 交付打包建议：请将目录 '{save_dir}' 下的模型权重与评估图表归档保存。")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args)
