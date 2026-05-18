# Task2: VisDrone 检测与视频多目标跟踪

## 1) 标注转换

```bash
python task2/prepare_visdrone.py \
  --visdrone_root /path/to/VisDrone2019-DET \
  --output_root datasets/visdrone_yolo
```

## 2) YOLOv8 训练

```bash
python task2/train_visdrone.py \
  --data datasets/visdrone_yolo/visdrone.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 960 \
  --batch 16 \
  --device 0 \
  --project outputs/task2_cloud \
  --name visdrone_yolov8n
```

## 3) 跟踪与越线计数

```bash
python task2/track_and_count.py \
  --model outputs/task2_cloud/visdrone_yolov8n/weights/best.pt \
  --source test_video.mp4 \
  --output_dir outputs/task2_cloud/tracking
```

## 4) 遮挡与 ID 跳变分析

```bash
python task2/analyze_id_switch.py \
  --track_csv outputs/task2_cloud/tracking/tracks.csv \
  --source_video test_video.mp4 \
  --output_dir outputs/task2_cloud/id_analysis_final \
  --num_frames 4 \
  --pick_mode crowded
```

## 5) 输出文件

- `outputs/task2_cloud/visdrone_yolov8n/final_metrics.json`
- `outputs/task2_cloud/tracking/test_video_tracked.mp4`
- `outputs/task2_cloud/tracking/tracking_summary.json`
- `outputs/task2_cloud/id_analysis_final/frame_*.jpg`
