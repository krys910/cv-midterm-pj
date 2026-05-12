# 计算机视觉期中 Project - Task 1

基于 `Oxford-IIIT Pet Dataset`，完成 ImageNet 预训练模型微调、超参数分析、预训练消融与注意力机制对比实验。

## 1. 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 数据准备

当前目录放置以下文件即可（你已下载完成）：

- `images.tar.gz`
- `annotations.tar.gz`

第一次运行训练时会自动解压为：

- `images/`
- `annotations/`

## 3. Baseline 微调（Task 1-1）

以 `resnet18` 为例，使用 ImageNet 预训练参数初始化，重置输出层并设置分层学习率（backbone 更小）：

```bash
python train.py \
  --data_root . \
  --output_dir outputs/baseline_resnet18 \
  --model resnet18 \
  --attention none \
  --init pretrained \
  --epochs 20 \
  --batch_size 32 \
  --base_lr 1e-4 \
  --head_lr 1e-3
```

训练产物：

- `outputs/.../best.pt`：验证集最佳权重
- `outputs/.../last.pt`：最后一个 epoch 权重
- `outputs/.../history.csv`：训练/验证曲线数据
- `outputs/.../metrics.json`：最终指标（best val acc、test acc 等）
- `outputs/.../args.json`：实验参数记录

## 4. 超参数分析（Task 1-2）

脚本会自动遍历不同模型与学习率组合：

```bash
python sweep.py \
  --data_root . \
  --output_root outputs \
  --epochs 20 \
  --models resnet18,resnet34 \
  --base_lrs 1e-4,3e-4 \
  --head_lrs 1e-3,3e-3
```

## 5. 预训练消融（Task 1-3）

在 `sweep.py` 中自动包含：

- `--init pretrained`
- `--init scratch`

可直接对比 `outputs/ablation/*/metrics.json` 的 `test_acc`。

## 6. 引入注意力机制（Task 1-4）

已实现 `SE` 与 `CBAM` 两种注意力模块，可在 ResNet backbone 的高层特征后插入：

```bash
python train.py \
  --data_root . \
  --output_dir outputs/attention_resnet18_se \
  --model resnet18 \
  --attention se \
  --init pretrained \
  --epochs 20 \
  --base_lr 1e-4 \
  --head_lr 1e-3
```

或使用批量脚本：

```bash
python sweep.py --data_root . --output_root outputs --with_attention
```

## 7. 可视化（wandb，可选）

默认会启用 wandb（已登录情况下会自动同步）。若你想临时关闭，使用 `--no_wandb`。

显式指定启用方式（可选）：

```bash
python train.py ... --use_wandb --wandb_project cv-midterm-pet
```

建议在报告中展示：

- train/val loss 曲线
- val accuracy 曲线
- 不同实验设置的最终 test accuracy 对比

如果已经训练完成、但当时没开 wandb，可把已有结果批量补录（无需重训）：

```bash
python log_to_wandb.py --output_root outputs/task1_full --project cv-midterm-pet
```

若还未登录 wandb，先执行：

```bash
wandb login
```

## 8. 报告建议结构（对应作业要求）

- 模型与数据集简介
- 实验设置：划分、网络结构、优化器、batch size、learning rate、epoch、loss、评估指标
- 结果对比：
  - Baseline
  - 超参数分析
  - 预训练消融（pretrained vs scratch）
  - 注意力模块（none vs SE vs CBAM）
- 结论与分析
- Github 仓库链接 + 模型权重下载链接

---

# Task 2: 场景目标检测与视频多目标跟踪

新增脚本目录：`task2/`

- `prepare_visdrone.py`：将 VisDrone DET 标注转为 YOLO 格式并生成 `visdrone.yaml`
- `train_visdrone.py`：YOLOv8 微调训练（默认同步 wandb）
- `track_and_count.py`：视频逐帧检测+跟踪（稳定 ID）并做越线计数
- `analyze_id_switch.py`：提取连续 3-4 帧，分析遮挡/ID 跳变
- `run_task2_pipeline.py`：一键串联以上流程

## Task2-1 数据预处理 + 微调训练（同步 wandb）

1) 准备数据（示例目录）

```text
VisDrone2019-DET/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

2) 转换标注

```bash
python task2/prepare_visdrone.py \
  --visdrone_root /path/to/VisDrone2019-DET \
  --output_root datasets/visdrone_yolo
```

3) 训练（默认 wandb 开启）

```bash
python task2/train_visdrone.py \
  --data datasets/visdrone_yolo/visdrone.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 960 \
  --batch 16 \
  --device mps \
  --project outputs/task2 \
  --name visdrone_yolov8n \
  --wandb_project cv-midterm-pet
```

若需要关闭 wandb：在命令后加 `--no_wandb`。

## Task2-2 视频流检测与多目标跟踪 + Task2-4 越线计数

```bash
python task2/track_and_count.py \
  --model outputs/task2/visdrone_yolov8n/weights/best.pt \
  --source /path/to/your_test_video.mp4 \
  --output_dir outputs/task2/tracking \
  --tracker bytetrack.yaml \
  --wandb_project cv-midterm-pet
```

可选指定虚拟计数线（`x1,y1,x2,y2`）：

```bash
--line 100,300,1200,300
```

输出：
- `outputs/task2/tracking/*_tracked.mp4`（带框、类别、Tracking ID、越线计数）
- `outputs/task2/tracking/tracks.csv`（逐帧跟踪结果）
- `outputs/task2/tracking/tracking_summary.json`

## Task2-3 遮挡与 ID 跳变分析

```bash
python task2/analyze_id_switch.py \
  --track_csv outputs/task2/tracking/tracks.csv \
  --source_video /path/to/your_test_video.mp4 \
  --output_dir outputs/task2/id_analysis \
  --num_frames 4 \
  --wandb_project cv-midterm-pet
```

输出：
- `outputs/task2/id_analysis/frame_*.jpg`（连续帧可视化）
- `outputs/task2/id_analysis/id_switch_report.json`（可能的 ID 跳变事件）

## 一键跑完整 Task2

```bash
python task2/run_task2_pipeline.py \
  --visdrone_root /path/to/VisDrone2019-DET \
  --test_video /path/to/your_test_video.mp4 \
  --workspace outputs/task2 \
  --epochs 80 \
  --device mps \
  --wandb_project cv-midterm-pet
```

