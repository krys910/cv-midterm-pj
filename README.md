# 计算机视觉期中项目（Task1-3）

本仓库包含课程期中项目三个任务的完整代码、训练流程与结果汇总脚本：

- `task1/`：ImageNet 预训练微调分类（Oxford-IIIT Pet）
- `task2/`：VisDrone 目标检测 + 视频多目标跟踪 + 越线计数
- `task3/`：从零实现 U-Net + 分割损失函数对比（CE / Dice / CE+Dice）

---

## 1. 仓库结构

```text
.
├── task1/
│   ├── train.py
│   ├── sweep.py
│   ├── summarize_results.py
│   └── src/
├── task2/
│   ├── prepare_visdrone.py
│   ├── train_visdrone.py
│   ├── track_and_count.py
│   └── analyze_id_switch.py
├── task3/
│   ├── train_unet.py
│   ├── run_loss_ablation.py
│   └── summarize_task3.py
├── outputs/                 # 训练输出（默认忽略，不提交）
└── generate_report_results.py
```

---

## 2. 环境配置

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

可选（仅 Task1 独立环境）：

```bash
pip install -r task1/requirements.txt
```

---

## 3. 数据准备

### Task1 / Task3（Oxford-IIIT Pet）

项目根目录放置：

- `images.tar.gz`
- `annotations.tar.gz`

首次训练会自动解压到 `images/` 与 `annotations/`。

### Task2（VisDrone）

准备如下目录（示例）：

```text
VisDrone2019-DET/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

---

## 4. Task1：分类微调（Baseline / Ablation / Attention）

### 4.1 单次训练

```bash
python task1/train.py \
  --data_root . \
  --output_dir outputs/task1_cloud_v2/baseline/resnet18_blr3e4_hlr3e3_s42 \
  --model resnet18 \
  --attention none \
  --init pretrained \
  --epochs 40 \
  --batch_size 32 \
  --base_lr 3e-4 \
  --head_lr 3e-3 \
  --weight_decay 5e-5 \
  --label_smoothing 0.05 \
  --dropout 0.1 \
  --seed 42
```

### 4.2 批量实验

```bash
python task1/sweep.py --data_root . --output_root task1/outputs/task1_full --with_attention
```

### 4.3 汇总结果

```bash
python task1/summarize_results.py --output_root outputs/task1_cloud_v2 --save_csv outputs/task1_cloud_v2/summary.csv
```

---

## 5. Task2：检测、跟踪与越线计数

### 5.1 标注转换（VisDrone -> YOLO）

```bash
python task2/prepare_visdrone.py \
  --visdrone_root /path/to/VisDrone2019-DET \
  --output_root datasets/visdrone_yolo
```

### 5.2 检测训练（YOLOv8）

```bash
python task2/train_visdrone.py \
  --data datasets/visdrone_yolo/visdrone.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 960 \
  --batch 16 \
  --device 0 \
  --project outputs/task2_cloud \
  --name visdrone_yolov8n \
  --wandb_project cv-midterm-pet
```

### 5.3 跟踪 + 越线计数

```bash
python task2/track_and_count.py \
  --model outputs/task2_cloud/visdrone_yolov8n/weights/best.pt \
  --source test_video.mp4 \
  --output_dir outputs/task2_cloud/tracking \
  --tracker bytetrack.yaml
```

### 5.4 遮挡/ID跳变分析（自动选片段）

```bash
python task2/analyze_id_switch.py \
  --track_csv outputs/task2_cloud/tracking/tracks.csv \
  --source_video test_video.mp4 \
  --output_dir outputs/task2_cloud/id_analysis_final \
  --num_frames 4 \
  --pick_mode crowded
```

---

## 6. Task3：U-Net 与损失函数对比

### 6.1 单次训练

```bash
python task3/train_unet.py \
  --data_root . \
  --output_dir outputs/task3_cloud/task3_unet_ce \
  --loss_type ce \
  --epochs 50 \
  --batch_size 16 \
  --image_size 256
```

`--loss_type` 可选：`ce`、`dice`、`ce_dice`

### 6.2 三种损失一键对比

```bash
python task3/run_loss_ablation.py \
  --data_root . \
  --output_root outputs/task3_cloud \
  --epochs 50 \
  --batch_size 16 \
  --image_size 256
```

### 6.3 汇总结果

```bash
python task3/summarize_task3.py --output_root outputs/task3_cloud
```

---

## 7. 报告结果自动汇总

将 Task1-3 结果汇总为可直接用于报告的 Markdown：

```bash
python generate_report_results.py --project_root . --output_md results_for_report.md
```

---

## 8. 结果产物说明

常见输出文件：

- `best.pt`：验证集最佳权重
- `last.pt`：最后 epoch 权重
- `history.csv/json`：训练曲线数据
- `metrics.json` / `final_metrics.json`：最终指标
- `tracking_summary.json`：越线计数与跟踪统计

---

## 9. 代码与模型链接（提交前请替换）

- Github Repo: <https://github.com/krys910/cv-midterm-pj>
- 模型权重（Google Drive）: `<替换为你的共享链接>`
- wandb 项目: `<替换为你的 wandb 项目链接>`

