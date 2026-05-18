# Task3: U-Net 与损失函数对比

## 1) 单次训练（示例：CE）

```bash
python task3/train_unet.py \
  --data_root . \
  --output_dir outputs/task3_cloud/task3_unet_ce \
  --loss_type ce \
  --epochs 50 \
  --batch_size 16 \
  --image_size 256
```

## 2) 三种损失函数对比（CE / Dice / CE+Dice）

```bash
python task3/run_loss_ablation.py \
  --data_root . \
  --output_root outputs/task3_cloud \
  --epochs 50 \
  --batch_size 16 \
  --image_size 256
```

## 3) 汇总结果

```bash
python task3/summarize_task3.py --output_root outputs/task3_cloud
```

## 4) 输出文件

- `outputs/task3_cloud/task3_unet_ce/final_metrics.json`
- `outputs/task3_cloud/task3_unet_dice/final_metrics.json`
- `outputs/task3_cloud/task3_unet_ce_dice/final_metrics.json`
- 各目录下 `history.json`、`best.pt`、`last.pt`
