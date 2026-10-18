# 深度学习股票趋势三分类实验运行说明

## 1. 创建虚拟环境

```bash
cd /Users/daiguorui/VSProject/dataset
python3 -m venv .venv
```

## 2. 安装依赖

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

如果网络慢，也可以直接安装核心依赖：

```bash
.venv/bin/python -m pip install torch scikit-learn matplotlib pandas numpy
```

## 3. 运行深度学习训练

```bash
.venv/bin/python train_deep_sequence_models.py
```

脚本会训练并对比 3 种端到端深度学习序列模型：

- LSTM
- GRU
- TCN

每种模型默认尝试两组学习率和训练轮次：

- learning_rate=0.001, epochs=6
- learning_rate=0.0005, epochs=10

## 4. 输出文件

训练完成后结果会写入：

```text
/Users/daiguorui/VSProject/dataset/deep_model_results
```

主要文件：

- `training_log.csv`：逐 epoch 训练日志
- `evaluation_metrics.csv`：所有模型配置的验证集/测试集指标对比
- `evaluation_report.md`：详细模型评估报告
- `best_model.pt`：验证集重点类别 F1 最优的模型权重
- `confusion_matrix_best.csv`：最佳模型测试集混淆矩阵
- `confusion_matrix_best.png`：最佳模型测试集混淆矩阵图片
- `predictions_test_*.npy`：各配置在测试集上的预测结果

## 5. 评估重点

脚本会输出：

- Accuracy
- Macro F1
- 下跌类 precision / recall / F1
- 震荡类 precision / recall / F1
- 拉伸类 precision / recall / F1
- 混淆矩阵
- `focus_macro_f1`：下跌类和拉伸类 F1 的均值

最佳模型按验证集 `focus_macro_f1` 选择，因为任务重点关注“下跌类”和“拉伸类”的识别能力。

## 6. 调整训练规模

如果训练太慢，可以在 `train_deep_sequence_models.py` 末尾的 `configs` 中减少配置或 epoch，例如只保留：

```python
configs = [
    Config("lstm", 0.001, 6),
    Config("gru", 0.001, 6),
    Config("tcn", 0.001, 6),
]
```

如果想提高效果，可以增加：

- `epochs`
- `hidden_size`
- `num_layers`
- 尝试更小学习率，如 `0.0003`
- 调整类别权重或采样策略以提高下跌类/拉伸类召回率
