# Ubuntu 训练环境快速启动指南 (NEU-DET 缺陷检测)

本工程已完成路径解耦与工业化训练逻辑封装，按以下四步即可在 Ubuntu 主机（配备 NVIDIA GPU）上完成高效训练与部署模型导出。

---

## 步骤 1：同步代码与数据集至 Ubuntu 主机

通过局域网 `rsync` 或 `scp` 将本项目完整拷贝到 Ubuntu 主机中（例如目录 `~/projects/NEU_DET_YOLOv8_Project`）：

```bash
# 在本地 Mac 终端执行同步命令（替换 your_user 和 your_ubuntu_ip）：
rsync -avzP --exclude '.git' \
  "/Users/youyi/Library/Mobile Documents/com~apple~CloudDocs/个人文件/盛相科技/07_开源项目与真实数据集/repositories/NEU_DET_YOLOv8_Project/" \
  your_user@your_ubuntu_ip:~/projects/NEU_DET_YOLOv8_Project/
```

> **提示**：如果网络不通，也可将 `NEU_DET_YOLOv8_Project` 文件夹压缩为 `.tar.gz` 或 `.zip` 拷至 U 盘或通过移动硬盘载入 Ubuntu。

---

## 步骤 2：在 Ubuntu 上一键创建 `.env` 虚拟环境 (免 Conda)

项目目录内已提供自动化环境构建脚本 `setup_env.sh`，直接利用 Python 原生 `venv` 模块在项目根目录下构建独立 `.env`：

```bash
cd ~/projects/NEU_DET_YOLOv8_Project

# 1. 确保系统包含基础 python3-venv (若已安装可跳过此命令)
sudo apt update && sudo apt install -y python3-venv python3-pip

# 2. 运行环境初始化脚本 (全自动创建 .env、安装 CUDA 版 PyTorch 与所有依赖)
bash setup_env.sh
```

脚本执行完毕后会自动在终端打印 GPU 识别状态与 CUDA 可用性。

---

## 步骤 3：一键启动训练与自动导出

我们为你提供了两种方式调用 `.env` 运行训练：

### 方式 A：使用快捷脚本直接运行（最省心）
```bash
cd ~/projects/NEU_DET_YOLOv8_Project

# 直接运行（默认参数：epochs=150, batch=16, imgsz=640, device=0）
./run_train.sh

# 自定义参数传参运行：./run_train.sh <epochs> <batch> <imgsz> <device>
./run_train.sh 200 16 640 0

# 后台挂起运行（推荐产线服务器做法）
nohup ./run_train.sh 150 16 640 0 > train.log 2>&1 &
tail -f train.log
```

### 方式 B：手动激活 `.env` 运行
```bash
cd ~/projects/NEU_DET_YOLOv8_Project
source .env/bin/activate

python3 train_industrial.py \
  --weights yolov8s.pt \
  --data data/data.yaml \
  --epochs 150 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

---

## 步骤 4：产出物与交付物验收

训练结束后，输出结果集中保存在 `runs/train_neu_det/yolov8s_baseline/` 目录下：

1. **`weights/best.pt`**：验证集 mAP 最优的 PyTorch 权重文件；
2. **`weights/best.onnx`**：已经简化且支持动态 Batch 的 ONNX 工业部署模型；
3. **`model_meta.json`**：部署交付规范文件，记录了输入尺寸、各缺陷类别的置信度阈值推荐与最终测试指标；
4. **`results.png` & `confusion_matrix.png`**：训练 Loss 下降曲线、混淆矩阵与 PR 曲线，用于直接向工艺/算法主管汇报。
