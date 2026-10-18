# 股票趋势三分类与量化交易

基于匿名化 33 维序列特征（60 时间步 × 33 通道）的股票未来趋势三分类预测，以及基于预测信号的量化交易策略设计与回测。

标签定义：`0 = 下跌类(fall)`，`1 = 震荡类(neutral)`，`2 = 拉伸类(pull)`。

## 项目结构

```
├── README.md
├── requirements.txt
├── data/                        # 数据说明（原始数据不入库）
├── docs/                        # 报告与汇报材料
│   ├── 股票趋势三分类与量化交易报告.md
│   ├── 任务三_策略绩效分析报告.md / .docx
│   ├── 任务三_数据对齐审计.md
│   ├── 任务一任务二_汇报.pptx
│   └── RUN_DEEP_LEARNING.md
├── src/
│   ├── task1_feature_analysis/  # 任务一：特征工程与数据分析
│   ├── task2_deep_models/       # 任务二：LSTM / GRU / TCN 深度序列模型
│   └── task3_strategy_backtest/ # 任务三：交易策略设计与回测
└── results/
    ├── task1_feature_analysis/  # 特征分析 CSV / 图表
    ├── task2_deep_models/       # 训练日志、评估指标、混淆矩阵
    └── task3_backtest/          # 2024-2025 真实行情回测结果（主策略 +105.66%）
```

## 任务一：特征工程与数据分析

`src/task1_feature_analysis/feature_engineering_analysis.py`

- 将每条 60×33 的序列压缩为 mean/std/min/max/first/last/delta/slope 聚合特征
- 分析类别分布、特征相关性、特征重要性（ANOVA / 随机森林 / 互信息）、KMeans 聚类结构
- 结果输出至 `results/task1_feature_analysis/`

```bash
python src/task1_feature_analysis/feature_engineering_analysis.py
```

## 任务二：深度序列模型三分类

`src/task2_deep_models/train_deep_sequence_models.py`

- 端到端训练并对比 LSTM / GRU / TCN
- 带类别权重的 CrossEntropyLoss + AdamW + CosineAnnealingLR
- 模型选择标准：验证集重点类别（下跌/拉伸）F1 均值 `focus_macro_f1`
- 最佳模型：TCN（test focus_macro_f1 = 0.4951）
- 详见 `docs/RUN_DEEP_LEARNING.md`

```bash
python src/task2_deep_models/train_deep_sequence_models.py
```

## 任务三：量化交易策略与回测（2024–2025 真实行情，主策略 +105.66%）

`src/task3_strategy_backtest/`

| 脚本 | 说明 |
|---|---|
| `task3_prepare_data.py` | 数据准备：把任务二 TCN 模型的测试集预测对齐到真实 (code, date)，生成信号与真实 OHLC 价格面板 |
| `task3_backtest_2024_2025.py` | 真实口径回测：次日开盘撮合、止损/止盈/最长持有、固定仓位、交易成本，主策略累计收益 +105.66% |
| `verify_task3_results.py` | 独立复核回测结果（账目恒等式、撮合、时点） |

运行流程：

```bash
python src/task3_strategy_backtest/task3_prepare_data.py
python src/task3_strategy_backtest/task3_backtest_2024_2025.py
python src/task3_strategy_backtest/verify_task3_results.py
```

结果输出至 `results/task3_backtest/real_2024_2025/`。
数据对齐审计结论见 `docs/任务三_数据对齐审计.md`。


## 环境与运行

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 数据

原始数据（`X_train.npy`、`test.csv`、行情 CSV 等）体积较大，不入库，请按 `data/README.md` 放置。
