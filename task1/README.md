# Task 1

Task1 相关代码已收纳在本目录：

- `train.py`：单次训练（baseline / 预训练消融 / 注意力）
- `sweep.py`：批量超参、消融、注意力实验
- `src/`：数据与模型实现
- `log_to_wandb.py`：补录历史实验到 wandb
- `summarize_results.py`：汇总实验结果
- `outputs/`：Task1 的训练输出

从项目根目录运行示例：

```bash
python task1/train.py --data_root . --output_dir task1/outputs/baseline_resnet18
```
