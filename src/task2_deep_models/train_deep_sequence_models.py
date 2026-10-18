"""Train end-to-end deep sequence models for stock trend classification.

Models compared:
    1. LSTM
    2. GRU
    3. TCN

Run after creating the virtual environment:
    .venv/bin/python train_deep_sequence_models.py

Outputs are written to ./deep_model_results:
    - training_log.csv
    - learning_step_analysis.csv
    - learning_curve_summary.csv
    - evaluation_metrics.csv
    - best_model.pt
    - confusion_matrix_*.csv
    - confusion_matrix_*.png
    - learning_curves_*.png
    - confusion_matrix_best.png
    - predictions_test_*.npy
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import csv
import json
import os
import random
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams['font.family'] = 'Heiti TC'
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deep_model_results"
CLASS_NAMES = ["下跌类/fall", "震荡类/neutral", "拉伸类/pull"]
FOCUS_CLASSES = [0, 2]
SEED = 20260920


@dataclass(frozen=True)
class Config:
    model: str
    learning_rate: float
    epochs: int
    batch_size: int = 32
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.15
    weight_decay: float = 1e-4


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.rnn(x)
        return self.head(hidden[-1])


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        return self.head(hidden[-1])


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.downsample(x))


class TCNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            TemporalBlock(input_size, hidden_size, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.network(x)[:, :, -1]
        return self.head(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    return torch.device("mps" if torch.mps.is_available() else "cpu")


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.load(ROOT / "X_train.npy").astype(np.float32)
    X_val = np.load(ROOT / "X_val.npy").astype(np.float32)
    X_test = np.load(ROOT / "X_test.npy").astype(np.float32)
    y_train = np.load(ROOT / "y_train.npy").astype(np.int64)
    y_val = np.load(ROOT / "y_val.npy").astype(np.int64)
    y_test = np.load(ROOT / "y_test.npy").astype(np.int64)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, y_train, (X_val - mean) / std, y_val, (X_test - mean) / std, y_test


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def build_model(config: Config, input_size: int) -> nn.Module:
    if config.model == "lstm":
        return LSTMClassifier(input_size, config.hidden_size, config.num_layers, config.dropout)
    if config.model == "gru":
        return GRUClassifier(input_size, config.hidden_size, config.num_layers, config.dropout)
    if config.model == "tcn":
        return TCNClassifier(input_size, config.hidden_size, config.dropout)
    raise ValueError(f"Unknown model: {config.model}")


def class_weights(y: np.ndarray, run_device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(np.float32)
    weights = len(y) / (len(counts) * counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=run_device)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, run_device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    preds = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    n = 0
    for xb, yb in loader:
        xb = xb.to(run_device)
        yb = yb.to(run_device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * len(yb)
        n += len(yb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), total_loss / max(n, 1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "focus_macro_f1": float(np.mean([report[CLASS_NAMES[i]]["f1-score"] for i in FOCUS_CLASSES])),
        "report": report,
        "confusion_matrix": cm,
    }


def train_one(
    config: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_device: torch.device,
) -> tuple[dict, list[dict], dict, dict]:
    set_seed(SEED)
    train_loader = make_loader(X_train, y_train, config.batch_size, True)
    train_eval_loader = make_loader(X_train, y_train, config.batch_size, False)
    val_loader = make_loader(X_val, y_val, config.batch_size, False)
    test_loader = make_loader(X_test, y_test, config.batch_size, False)
    model = build_model(config, X_train.shape[2]).to(run_device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, run_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    epoch_rows = []
    best_state = None
    best_val = -1.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(run_device)
            yb = yb.to(run_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(yb)
            seen += len(yb)
        scheduler.step()
        train_pred, _ = predict(model, train_eval_loader, run_device)
        val_pred, val_loss = predict(model, val_loader, run_device)
        train_metrics = metrics(y_train, train_pred)
        val_metrics = metrics(y_val, val_pred)
        row = {
            **asdict(config),
            "epoch": epoch,
            "train_loss": train_loss / seen,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_focus_f1": val_metrics["focus_macro_f1"],
            "val_f1_fall": val_metrics["report"][CLASS_NAMES[0]]["f1-score"],
            "val_f1_pull": val_metrics["report"][CLASS_NAMES[2]]["f1-score"],
        }
        epoch_rows.append(row)
        print(
            f"{config.model} lr={config.learning_rate} ep={epoch}/{config.epochs} "
            f"val_acc={row['val_accuracy']:.4f} val_macro_f1={row['val_macro_f1']:.4f} focus_f1={row['val_focus_f1']:.4f}",
            flush=True,
        )
        if val_metrics["focus_macro_f1"] > best_val:
            best_val = val_metrics["focus_macro_f1"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    val_pred, _ = predict(model, val_loader, run_device)
    test_pred, _ = predict(model, test_loader, run_device)
    val_metrics = metrics(y_val, val_pred)
    test_metrics = metrics(y_test, test_pred)
    return {"state_dict": best_state, "config": asdict(config)}, epoch_rows, val_metrics, {**test_metrics, "pred": test_pred}


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name, *row.tolist()])


def plot_confusion(path: Path, matrix: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(3), CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(3), CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def experiment_name(row: dict) -> str:
    return f"{row['model']}_lr{row['learning_rate']}_ep{row['epochs']}"


def plot_learning_curves(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(experiment_name(row), []).append(row)

    curve_specs = [
        ("val_focus_f1", "Validation focus F1 (fall + pull)", OUT / "learning_curves_val_focus_f1.png"),
        ("val_macro_f1", "Validation macro F1", OUT / "learning_curves_val_macro_f1.png"),
        ("val_accuracy", "Validation accuracy", OUT / "learning_curves_val_accuracy.png"),
    ]
    for metric, title, path in curve_specs:
        fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
        for name, items in groups.items():
            items = sorted(items, key=lambda item: item["epoch"])
            ax.plot([item["epoch"] for item in items], [item[metric] for item in items], marker="o", linewidth=1.6, markersize=3, label=name)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    for name, items in groups.items():
        items = sorted(items, key=lambda item: item["epoch"])
        ax.plot([item["epoch"] for item in items], [item["train_loss"] for item in items], linestyle="-", linewidth=1.5, label=f"{name} train")
        ax.plot([item["epoch"] for item in items], [item["val_loss"] for item in items], linestyle="--", linewidth=1.5, label=f"{name} val")
    ax.set_title("Training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves_loss.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    for name, items in groups.items():
        items = sorted(items, key=lambda item: item["epoch"])
        ax.plot([item["epoch"] for item in items], [item["val_f1_fall"] for item in items], linewidth=1.4, label=f"{name} fall")
        ax.plot([item["epoch"] for item in items], [item["val_f1_pull"] for item in items], linestyle="--", linewidth=1.4, label=f"{name} pull")
    ax.set_title("Validation F1 by focus class")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves_focus_classes.png")
    plt.close(fig)


def analyze_learning_steps(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(experiment_name(row), []).append(row)

    step_rows = []
    summary_rows = []
    for name, items in sorted(groups.items()):
        items = sorted(items, key=lambda item: item["epoch"])
        previous = None
        for item in items:
            step_row = {
                "experiment": name,
                "model": item["model"],
                "learning_rate": item["learning_rate"],
                "epochs": item["epochs"],
                "epoch": item["epoch"],
                "train_loss": item["train_loss"],
                "val_loss": item["val_loss"],
                "train_macro_f1": item["train_macro_f1"],
                "val_macro_f1": item["val_macro_f1"],
                "val_focus_f1": item["val_focus_f1"],
                "val_f1_fall": item["val_f1_fall"],
                "val_f1_pull": item["val_f1_pull"],
                "delta_val_focus_f1": 0.0 if previous is None else item["val_focus_f1"] - previous["val_focus_f1"],
                "delta_val_macro_f1": 0.0 if previous is None else item["val_macro_f1"] - previous["val_macro_f1"],
                "delta_val_loss": 0.0 if previous is None else item["val_loss"] - previous["val_loss"],
                "generalization_gap_macro_f1": item["train_macro_f1"] - item["val_macro_f1"],
            }
            step_rows.append(step_row)
            previous = item

        best_focus = max(items, key=lambda item: item["val_focus_f1"])
        best_macro = max(items, key=lambda item: item["val_macro_f1"])
        best_acc = max(items, key=lambda item: item["val_accuracy"])
        first = items[0]
        last = items[-1]
        post_best_items = [item for item in items if item["epoch"] >= best_focus["epoch"]]
        min_post_best_loss = min(item["val_loss"] for item in post_best_items)
        overfit_flag = (last["val_loss"] > min_post_best_loss + 0.01) and (last["train_loss"] < best_focus["train_loss"])
        summary_rows.append(
            {
                "experiment": name,
                "model": best_focus["model"],
                "learning_rate": best_focus["learning_rate"],
                "epochs": best_focus["epochs"],
                "best_focus_epoch": best_focus["epoch"],
                "best_val_focus_f1": best_focus["val_focus_f1"],
                "best_val_macro_f1": best_macro["val_macro_f1"],
                "best_macro_epoch": best_macro["epoch"],
                "best_val_accuracy": best_acc["val_accuracy"],
                "best_accuracy_epoch": best_acc["epoch"],
                "first_val_focus_f1": first["val_focus_f1"],
                "last_val_focus_f1": last["val_focus_f1"],
                "focus_f1_gain_first_to_best": best_focus["val_focus_f1"] - first["val_focus_f1"],
                "focus_f1_change_best_to_last": last["val_focus_f1"] - best_focus["val_focus_f1"],
                "last_generalization_gap_macro_f1": last["train_macro_f1"] - last["val_macro_f1"],
                "overfit_warning": overfit_flag,
            }
        )
    return step_rows, summary_rows


def make_report(summary_rows: list[dict], learning_summary_rows: list[dict], best_name: str, best_metrics: dict, data_info: dict) -> str:
    lines = [
        "# 股票未来趋势三分类深度学习模型评估报告",
        "",
        "## 数据集",
        f"- train={data_info['train']}，val={data_info['val']}，test={data_info['test']}，序列长度={data_info['time_steps']}，特征维度={data_info['channels']}。",
        "- 类别：0=下跌类，1=震荡类，2=拉伸类。",
        "- 重点指标：下跌类与拉伸类的 precision、recall、F1，以及二者 F1 均值 focus_macro_f1。",
        "",
        "## 模型与训练",
        "- 模型：LSTM、GRU、TCN，均为端到端 PyTorch 深度学习模型。",
        "- 每个模型使用学习率 lr=0.001，训练 20 轮。",
        "- 损失函数：带类别权重的 CrossEntropyLoss；优化器：AdamW；学习率调度：CosineAnnealingLR。",
        "- 最佳模型选择标准：验证集 focus_macro_f1 最大。",
        "",
        "## 模型对比",
        "| 模型 | lr | epochs | val_acc | val_macro_f1 | val_focus_f1 | val_f1_下跌 | val_f1_拉伸 | test_acc | test_macro_f1 | test_focus_f1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['learning_rate']} | {row['epochs']} | "
            f"{row['val_accuracy']:.4f} | {row['val_macro_f1']:.4f} | {row['val_focus_f1']:.4f} | "
            f"{row['val_f1_fall']:.4f} | {row['val_f1_pull']:.4f} | "
            f"{row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} | {row['test_focus_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 学习过程对比",
            "| 模型配置 | 最佳focus epoch | best val_focus_f1 | first->best提升 | best->last变化 | last泛化差距 | 过拟合提示 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in learning_summary_rows:
        lines.append(
            f"| {row['experiment']} | {row['best_focus_epoch']} | {row['best_val_focus_f1']:.4f} | "
            f"{row['focus_f1_gain_first_to_best']:.4f} | {row['focus_f1_change_best_to_last']:.4f} | "
            f"{row['last_generalization_gap_macro_f1']:.4f} | {'是' if row['overfit_warning'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "学习曲线图已输出：",
            "- `learning_curves_val_focus_f1.png`：重点类别 F1 随 epoch 变化。",
            "- `learning_curves_val_macro_f1.png`：验证集 macro F1 随 epoch 变化。",
            "- `learning_curves_val_accuracy.png`：验证集准确率随 epoch 变化。",
            "- `learning_curves_loss.png`：训练/验证 loss 对比。",
            "- `learning_curves_focus_classes.png`：下跌类与拉伸类 F1 的逐 epoch 对比。",
        ]
    )
    lines.extend(
        [
            "",
            f"## 最佳模型：{best_name}",
            f"- Test Accuracy：{best_metrics['accuracy']:.4f}",
            f"- Test Macro F1：{best_metrics['macro_f1']:.4f}",
            f"- Test Focus Macro F1：{best_metrics['focus_macro_f1']:.4f}",
            "",
            "## 最佳模型测试集分类报告",
            "| 类别 | precision | recall | F1 | support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in CLASS_NAMES:
        item = best_metrics["report"][name]
        lines.append(f"| {name} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1-score']:.4f} | {int(item['support'])} |")
    lines.extend(["", "## 最佳模型测试集混淆矩阵", "| true \\ pred | 下跌类 | 震荡类 | 拉伸类 |", "|---|---:|---:|---:|"])
    for name, row in zip(CLASS_NAMES, best_metrics["confusion_matrix"]):
        lines.append(f"| {name} | {int(row[0])} | {int(row[1])} | {int(row[2])} |")
    lines.extend(
        [
            "",
            "## 结论",
            "- 报告优先比较重点类别的 F1，而不是只看整体准确率。",
            "- 下跌类 recall 代表风险样本找回能力；拉伸类 recall 代表机会样本捕捉能力。",
            "- 若业务更看重拉伸或下跌的召回率，可进一步调高对应类别权重或调整分类阈值/采样策略。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deep sequence models and export result tables/plots.")
    parser.add_argument("--report", action="store_true", help="also write evaluation_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(exist_ok=True)
    set_seed(SEED)
    run_device = device()
    start = time.time()
    print(f"device={run_device}", flush=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_arrays()
    configs = [
        Config("lstm", 0.001, 20),
        Config("gru", 0.001, 20),
        Config("tcn", 0.001, 20),
        # Config("lstm", 0.0005, 20),
        # Config("gru", 0.0005, 20),
        # Config("tcn", 0.0005, 20),
    ]
    all_epoch_rows = []
    summary_rows = []
    best = None
    for config in configs:
        print(f"training {config}", flush=True)
        checkpoint, epoch_rows, val_metrics, test_metrics = train_one(config, X_train, y_train, X_val, y_val, X_test, y_test, run_device)
        all_epoch_rows.extend(epoch_rows)
        name = f"{config.model}_lr{config.learning_rate}_ep{config.epochs}"
        np.save(OUT / f"predictions_test_{name}.npy", test_metrics["pred"])
        write_confusion(OUT / f"confusion_matrix_test_{name}.csv", test_metrics["confusion_matrix"])
        plot_confusion(OUT / f"confusion_matrix_test_{name}.png", test_metrics["confusion_matrix"], f"Test confusion matrix: {name}")
        row = {
            **asdict(config),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_focus_f1": val_metrics["focus_macro_f1"],
            "val_f1_fall": val_metrics["report"][CLASS_NAMES[0]]["f1-score"],
            "val_f1_pull": val_metrics["report"][CLASS_NAMES[2]]["f1-score"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_focus_f1": test_metrics["focus_macro_f1"],
            "test_f1_fall": test_metrics["report"][CLASS_NAMES[0]]["f1-score"],
            "test_f1_pull": test_metrics["report"][CLASS_NAMES[2]]["f1-score"],
        }
        summary_rows.append(row)
        candidate = (val_metrics["focus_macro_f1"], val_metrics["macro_f1"], name, checkpoint, test_metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    _, _, best_name, best_checkpoint, best_metrics = best
    write_rows(OUT / "training_log.csv", all_epoch_rows)
    write_rows(OUT / "evaluation_metrics.csv", summary_rows)
    learning_step_rows, learning_summary_rows = analyze_learning_steps(all_epoch_rows)
    write_rows(OUT / "learning_step_analysis.csv", learning_step_rows)
    write_rows(OUT / "learning_curve_summary.csv", learning_summary_rows)
    plot_learning_curves(all_epoch_rows)
    torch.save(best_checkpoint, OUT / "best_model.pt")
    plot_confusion(OUT / "confusion_matrix_best.png", best_metrics["confusion_matrix"], f"Best model: {best_name}")
    write_confusion(OUT / "confusion_matrix_best.csv", best_metrics["confusion_matrix"])
    data_info = {
        "train": len(y_train),
        "val": len(y_val),
        "test": len(y_test),
        "time_steps": X_train.shape[1],
        "channels": X_train.shape[2],
    }
    if args.report:
        (OUT / "evaluation_report.md").write_text(make_report(summary_rows, learning_summary_rows, best_name, best_metrics, data_info), encoding="utf-8")
    metadata = {
        "best_model": best_name,
        "device": str(run_device),
        "runtime_seconds": round(time.time() - start, 2),
        "data": data_info,
        "torch_version": torch.__version__,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best_model={best_name}")
    print(f"test_accuracy={best_metrics['accuracy']:.4f} test_macro_f1={best_metrics['macro_f1']:.4f} test_focus_f1={best_metrics['focus_macro_f1']:.4f}")
    if args.report:
        print(f"report={OUT / 'evaluation_report.md'}")
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
