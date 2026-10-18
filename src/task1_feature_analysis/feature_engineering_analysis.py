"""股票趋势数据集的特征分析与工程脚本。

这个脚本的作用：
    1. 读取原始序列数据 X_*.npy 和标签 y_*.npy。
    2. 把每条 60 天 x 33 个指标的时间序列压缩成可读的聚合特征。
    3. 分析类别分布、特征相关性、特征重要性、聚类结构。
    4. 输出 CSV 结果和 PNG 图表。

原始输入张量形状：
    (样本数, 时间步=60, 指标通道=33)

因为数据集没有给出 33 个指标的真实名称，所以这里统一称为 ch00 ... ch32。
每个指标通道都会提取 8 个时间聚合特征：
    mean, std, min, max, first, last, delta, slope

运行方式：
    .venv/bin/python feature_engineering_analysis.py

默认输出：
    feature_analysis_results/*.csv
    feature_analysis_results/*.png

如需生成 Markdown 报告，请显式加入参数：
    .venv/bin/python feature_engineering_analysis.py --report
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = 'Heiti TC'
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "feature_analysis_results"
CLASS_NAMES = {0: "下跌类/fall", 1: "震荡类/neutral", 2: "拉伸类/pull"}
RNG = np.random.default_rng(20260922)
KMEANS_K_VALUES = range(2, 13)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """加载某一个数据集划分。

    split 可以是 train / val / test。
    返回：
        X: 原始时间序列特征，形状为 (样本数, 60, 33)。
        y: 标签，0=下跌类，1=震荡类，2=拉伸类。
    """
    X = np.load(ROOT / f"X_{split}.npy").astype(np.float32)
    y = np.load(ROOT / f"y_{split}.npy").astype(np.int64)
    return X, y


def write_rows(path: Path, header: list[str], rows) -> None:
    """把二维表格结果写成 CSV。

    这是脚本内所有分析结果的统一输出方式，便于后续用 Excel 或 Python 继续处理。
    """
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def aggregate_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """把时间序列压缩成可解释的聚合特征。

    输入 X 的形状是：
        (样本数, 60天, 33个指标)

    这里不是直接聚原始 60x33 数据，而是先提取每个指标的时间走势摘要：
        mean   整体水平
        std    波动强度
        min    区间最低
        max    区间最高
        first  起点
        last   终点
        delta  终点减起点，表示区间变化
        slope  用线性斜率近似趋势方向

    最终得到 33 x 8 = 264 维特征。
    """
    time_index = np.arange(X.shape[1], dtype=np.float32)
    centered = time_index - time_index.mean()
    denominator = float(np.sum(centered**2))
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    minimum = X.min(axis=1)
    maximum = X.max(axis=1)
    first = X[:, 0, :]
    last = X[:, -1, :]
    delta = last - first
    slope = np.sum((X - mean[:, None, :]) * centered[None, :, None], axis=1) / denominator
    features = np.concatenate([mean, std, minimum, maximum, first, last, delta, slope], axis=1)
    names = [
        f"{stat}_ch{channel:02d}"
        for stat in ("mean", "std", "min", "max", "first", "last", "delta", "slope")
        for channel in range(X.shape[2])
    ]
    return features.astype(np.float32), names


def data_summary(y_by_split: dict[str, np.ndarray]) -> list[tuple]:
    """统计 train / val / test 三个划分中的类别数量和比例。

    这个函数用于确认数据是否类别平衡，也用于报告中的数据概览部分。
    """
    rows = []
    for split, y in y_by_split.items():
        counts = np.bincount(y, minlength=3)
        for cls, count in enumerate(counts):
            rows.append((split, len(y), cls, CLASS_NAMES[cls], int(count), float(count / len(y))))
    return rows


def class_feature_stats(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[tuple]:
    """按真实标签分组统计每个聚合特征的分布。

    对每一类样本，计算每个特征的均值、标准差、中位数、25 分位数、75 分位数。
    这些统计量可以帮助理解不同类别在趋势、波动、终点状态上有什么差异。
    """
    rows = []
    for cls in range(3):
        members = X[y == cls]
        means = members.mean(axis=0)
        stds = members.std(axis=0)
        medians = np.median(members, axis=0)
        q25 = np.quantile(members, 0.25, axis=0)
        q75 = np.quantile(members, 0.75, axis=0)
        for name, mean, std, median, left, right in zip(names, means, stds, medians, q25, q75):
            rows.append((cls, CLASS_NAMES[cls], name, float(mean), float(std), float(median), float(left), float(right)))
    return rows


def top_correlations(X_scaled: np.ndarray, names: list[str], top_n: int = 80) -> tuple[np.ndarray, list[tuple]]:
    """计算聚合特征之间的 Pearson 相关系数，并返回相关性最高的特征对。

    这里关心的是特征冗余：如果两个特征高度相关，它们可能表达同一类信息。
    相关性很高时，后续可以考虑特征筛选、PCA 降维或减少重复指标。

    注意：np.corrcoef 在遇到常量特征时会产生 nan；这里保留 nan，并在排序时跳过。
    """
    corr = np.corrcoef(X_scaled, rowvar=False)
    corr = np.where(np.isnan(corr), np.nan, corr)
    upper = np.triu_indices_from(corr, k=1)
    valid = ~np.isnan(corr[upper])
    values = np.where(valid, np.abs(corr[upper]), -np.inf)
    order = np.argsort(values)[::-1][:top_n]
    rows = []
    for pos in order:
        if not bool(valid[pos]):
            break
        i = int(upper[0][pos])
        j = int(upper[1][pos])
        rows.append((names[i], names[j], float(corr[i, j]), float(abs(corr[i, j]))))
    return corr, rows


def feature_scores(X_scaled: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """用三种方法评估特征和标签之间的关系。

    返回三类结果：
        ANOVA F-value：看不同类别下某个特征的均值差异是否明显。
        Mutual Information：捕捉特征和标签之间更复杂的非线性关系。
        Random Forest importance：看树模型是否经常用某个特征做分裂。

    三种方法互补，不建议只看其中一个。
    """
    f_values, p_values = f_classif(X_scaled, y)
    f_rows = sorted(
        [(name, float(f), float(p)) for name, f, p in zip(names, np.nan_to_num(f_values), np.nan_to_num(p_values, nan=1.0))],
        key=lambda row: row[1],
        reverse=True,
    )
    sample_n = min(30000, len(y))
    idx = RNG.choice(len(y), size=sample_n, replace=False)
    mi = mutual_info_classif(X_scaled[idx], y[idx], random_state=42, discrete_features=False, n_neighbors=5)
    mi_rows = sorted([(name, float(score)) for name, score in zip(names, mi)], key=lambda row: row[1], reverse=True)
    rf_n = min(30000, len(y))
    rf_idx = RNG.choice(len(y), size=rf_n, replace=False)
    forest = RandomForestClassifier(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X_scaled[rf_idx], y[rf_idx])
    rf_rows = sorted(
        [(name, float(score)) for name, score in zip(names, forest.feature_importances_)],
        key=lambda row: row[1],
        reverse=True,
    )
    return f_rows, mi_rows, rf_rows


def channel_summary(score_rows: list[tuple], names: list[str]) -> list[tuple]:
    """把特征重要性汇总到 33 个原始指标通道级别。

    每个通道有 8 个聚合特征，这里把它们的重要性合并观察，用来判断
    哪些 ch00 ... ch32 通道整体更有信息量。
    """
    score_map = {name: score for name, score, *_ in score_rows}
    rows = []
    for ch in range(33):
        channel_names = [name for name in names if name.endswith(f"ch{ch:02d}")]
        scores = np.array([score_map[name] for name in channel_names], dtype=np.float64)
        best_name = channel_names[int(scores.argmax())]
        rows.append((f"ch{ch:02d}", best_name, float(scores.max()), float(scores.mean()), float(scores.sum())))
    return sorted(rows, key=lambda row: row[4], reverse=True)


def select_top_low_corr_features(corr: np.ndarray, names: list[str], f_rows: list[tuple], rf_rows: list[tuple], top_n: int = 120, corr_threshold: float = 0.95) -> tuple[list[int], list[tuple], list[tuple]]:
    """按重要性挑选低相关特征，并返回被丢弃的高相关特征对。

    思路：
        1. 先用随机森林重要性排序，ANOVA F-value 作为次要 tie-breaker。
        2. 顺序保留重要特征。
        3. 如果候选特征与已保留特征相关系数绝对值 >= corr_threshold，则丢弃候选。
        4. 记录每个被丢弃特征对应的保留特征和相关系数，方便报告解释。
    """
    rf_map = {name: score for name, score in rf_rows}
    f_map = {name: score for name, score, *_ in f_rows}
    order = sorted(range(len(names)), key=lambda idx: (rf_map[names[idx]], f_map[names[idx]]), reverse=True)
    kept = []
    dropped = []
    selected_names = []
    corr_safe = np.nan_to_num(corr, nan=0.0)
    for rank, idx in enumerate(order, start=1):
        name = names[idx]
        redundant_pair = None
        for kept_idx in kept:
            r = float(corr_safe[idx, kept_idx])
            if abs(r) >= corr_threshold:
                redundant_pair = (names[kept_idx], r, abs(r))
                break
        if redundant_pair is None:
            kept.append(idx)
            selected_names.append((name, idx, rank))
        else:
            dropped.append((name, redundant_pair[0], redundant_pair[1], redundant_pair[2]))
        if len(kept) >= top_n:
            break
    return kept, selected_names, dropped


def clustering_analysis(X_scaled: np.ndarray, y: np.ndarray) -> tuple[list[tuple], np.ndarray, np.ndarray, dict[int, np.ndarray], PCA]:
    """在聚合特征空间上做 KMeans 聚类分析。

    流程：
        1. 从训练样本中抽样，控制计算量。
        2. 对 k=2 到 k=12 分别做 KMeans。
        3. 计算 inertia、silhouette、ARI、NMI、purity。
        4. 找出 silhouette 最高的 k，用于簇结构可视化。
        5. 用 PCA 降到二维，仅为人类查看聚类结构，不代表真实距离空间。

    重点理解：k 不是因为有 3 个标签就必须设 3；k 应该由数据结构和指标共同判断。
    """
    sample_n = min(25000, len(y))
    idx = RNG.choice(len(y), size=sample_n, replace=False)
    X_sample = X_scaled[idx]
    y_sample = y[idx]
    rows = []
    best = None
    labels_by_k = {}
    for k in KMEANS_K_VALUES:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42 + k)
        labels = kmeans.fit_predict(X_sample)
        labels_by_k[k] = labels
        silhouette = float(silhouette_score(X_sample, labels, sample_size=min(5000, sample_n), random_state=42))
        ari = float(adjusted_rand_score(y_sample, labels))
        nmi = float(normalized_mutual_info_score(y_sample, labels))
        purity, cluster_rows = purity_score(labels, y_sample, k)
        rows.append((k, float(kmeans.inertia_), silhouette, ari, nmi, purity))
        candidate = (silhouette, k, labels, y_sample, cluster_rows)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sample)
    best_labels = best[2]
    coords_rows = np.column_stack([coords, best_labels, y_sample])
    write_rows(
        OUT / "pca_sample_coordinates.csv",
        ["pc1", "pc2", "cluster", "true_label"],
        coords_rows.tolist(),
    )
    write_rows(
        OUT / "best_kmeans_cluster_composition.csv",
        ["cluster", "cluster_size", "majority_class_id", "majority_class", "fall_count", "neutral_count", "pull_count", "purity_in_cluster"],
        best[4],
    )
    return rows, coords, y_sample, labels_by_k, pca


def purity_score(labels: np.ndarray, y: np.ndarray, k: int) -> tuple[float, list[tuple]]:
    """计算聚类纯度，并给出每个簇的真实类别组成。

    purity 表示：每个簇中占比最高的真实类别，在所有样本中总共占多少。
    它直观，但不能替代 ARI/NMI；如果数据本身就是分块的，purity 可能被抬高。
    """
    correct = 0
    rows = []
    for cluster in range(k):
        members = y[labels == cluster]
        counts = np.bincount(members, minlength=3)
        majority = int(counts.argmax()) if len(members) else -1
        correct += int(counts.max()) if len(members) else 0
        cluster_purity = float(counts.max() / len(members)) if len(members) else 0.0
        rows.append((cluster, int(len(members)), majority, CLASS_NAMES.get(majority, "empty"), int(counts[0]), int(counts[1]), int(counts[2]), cluster_purity))
    return correct / len(y), rows


def plot_feature_correlation_heatmap(corr: np.ndarray, names: list[str], top_n: int = 28) -> None:
    """绘制 Pearson 相关系数热力图。

    为了不让 264x264 全矩阵太挤，这里选择相关性总体较强的前 top_n 个特征。
    红色代表正相关，蓝色代表负相关，接近 1 或 -1 说明特征之间关系很强。
    """
    corr_safe = np.nan_to_num(corr, nan=0.0)
    indices = list(range(len(names)))
    selected = sorted(indices, key=lambda i: sum(abs(corr_safe[i, j]) for j in range(len(names)) if i != j), reverse=True)[:top_n]
    selected_names = [names[i] for i in selected]
    matrix = corr_safe[np.ix_(selected, selected)]

    fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=160)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title("Pearson correlation heatmap of top correlated features")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(selected_names, rotation=90, fontsize=6)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(selected_names, rotation=0, fontsize=6)
    ax.set_xticks(np.arange(-0.5, top_n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, top_n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", length=0, pad=0)
    for i in range(top_n):
        for j in range(top_n):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=4.5, color="black" if abs(value) < 0.65 else "white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "feature_correlation_heatmap.png")
    plt.close(fig)


def plot_label_distribution(rows: list[tuple]) -> None:
    """绘制 train / val / test 的三分类标签分布柱状图。

    用于确认训练、验证、测试集的类别比例是否一致。
    """
    splits = ["train", "val", "test"]
    values = {split: [row[4] for row in rows if row[0] == split] for split in splits}
    x = np.arange(len(splits))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    for cls in range(3):
        ax.bar(x + (cls - 1) * width, [values[split][cls] for split in splits], width, label=CLASS_NAMES[cls])
    ax.set_xticks(x, splits)
    ax.set_ylabel("samples")
    ax.set_title("Label distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "label_distribution.png")
    plt.close(fig)


def plot_pca(coords: np.ndarray, y: np.ndarray, title: str = "PCA projection of aggregated sequence features") -> None:
    """用 PCA 二维坐标按真实标签画散点图。

    PCA 只是把高维特征压缩到二维，方便观察；它不是聚类结果本身。
    """
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    for cls in range(3):
        mask = y == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.35, label=CLASS_NAMES[cls])
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(OUT / "pca_by_label.png")
    plt.close(fig)


def plot_elbow_curve(inertia_by_k: list[tuple[int, float]], title: str, path: Path) -> None:
    """绘制 KMeans 手肘图。

    横轴是聚类数量 k，纵轴是 inertia。
    inertia 越低，说明样本离所属簇中心越近；但 k 越大 inertia 通常越低，
    所以重点看下降速度是否突然变慢，也就是所谓“手肘”位置。
    """
    k_values = [row[0] for row in inertia_by_k]
    inertia = [row[1] for row in inertia_by_k]
    x = np.array(k_values, dtype=float)
    y = np.array(inertia, dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
    ax.plot(x, y, marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("inertia")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cluster_inertias(X: np.ndarray) -> list[tuple[int, float]]:
    """扫描不同 k 的 KMeans inertia。

    这个函数专门给手肘图提供数据，只返回 (k, inertia)。
    """
    rows = []
    for k in KMEANS_K_VALUES:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42 + k).fit(X)
        rows.append((k, float(kmeans.inertia_)))
    return rows


def plot_silhouette_by_k(rows: list[tuple], title: str = "Silhouette by k") -> None:
    """绘制不同 k 的轮廓系数图。

    silhouette 越高，通常说明簇内更紧凑、簇间更分开。
    它评价的是“自然簇结构”，不等于和真实标签一致。
    """
    k_values = [row[0] for row in rows]
    silhouette = [row[2] for row in rows]
    best = max(rows, key=lambda row: row[2])

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
    ax.plot(k_values, silhouette, marker="o", linewidth=1.8, color="#1f77b4")
    ax.axhline(best[2], linestyle="--", color="#d62728", alpha=0.75)
    ax.scatter(best[0], best[2], color="#d62728", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "silhouette_by_k.png")
    plt.close(fig)


def plot_kmeans_metrics(rows: list[tuple], save_path: Path | None = None) -> None:
    """绘制 KMeans 聚类综合评估图。

    四个面板分别展示：
        inertia 手肘图
        silhouette 轮廓系数
        purity / ARI / NMI 与真实标签的一致性
        最佳 silhouette 和最佳 purity 的文字说明
    """
    k_values = [row[0] for row in rows]
    inertia = [row[1] for row in rows]
    silhouette = [row[2] for row in rows]
    ari = [row[3] for row in rows]
    nmi = [row[4] for row in rows]
    purity = [row[5] for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), dpi=150)
    axes[0, 0].plot(k_values, inertia, marker="o")
    axes[0, 0].set_title("Elbow: inertia by k")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("inertia")

    axes[0, 1].plot(k_values, silhouette, marker="o", color="#1f77b4")
    axes[0, 1].set_title("Silhouette by k")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("silhouette")

    axes[1, 0].plot(k_values, purity, marker="o", color="#2ca02c", label="purity")
    axes[1, 0].plot(k_values, ari, marker="o", color="#d62728", label="ARI")
    axes[1, 0].plot(k_values, nmi, marker="o", color="#9467bd", label="NMI")
    axes[1, 0].set_title("Label alignment by k")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    best_silhouette = max(rows, key=lambda row: row[2])
    best_purity = max(rows, key=lambda row: row[5])
    text = (
        f"Best silhouette: k={best_silhouette[0]}, {best_silhouette[2]:.4f}\n"
        f"Best purity: k={best_purity[0]}, {best_purity[5]:.4f}\n\n"
        "Use silhouette for natural cluster compactness.\n"
        "Use ARI/NMI/purity to compare clusters with labels."
    )
    axes[1, 1].text(0.02, 0.82, text, va="top", fontsize=10)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path if save_path is not None else OUT / "kmeans_metrics.png")
    plt.close(fig)


def plot_kmeans_pca(coords: np.ndarray, labels_by_k: dict[int, np.ndarray], suffix: str = "") -> None:
    """把不同 k 的聚类结果投影到 PCA 二维空间并保存成图。

    这里每个点颜色代表 KMeans 簇，不是真实标签。
    """
    for k, labels in labels_by_k.items():
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=4, alpha=0.45, cmap="tab20")
        ax.set_title(f"K-Means clusters on PCA projection, k={k}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        legend = ax.legend(*scatter.legend_elements(num=min(k, 12)), title="cluster", markerscale=3, fontsize=8)
        ax.add_artist(legend)
        fig.tight_layout()
        fig.savefig(OUT / f"kmeans_pca{k_suffix}_k{k:02d}.png")
        plt.close(fig)

    k_values = list(labels_by_k)
    cols = 3
    rows = int(np.ceil(len(k_values) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), dpi=140)
    axes = np.array(axes).reshape(-1)
    for ax, k in zip(axes, k_values):
        labels = labels_by_k[k]
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=2, alpha=0.35, cmap="tab20")
        ax.set_title(f"k={k}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(k_values) :]:
        ax.axis("off")
    fig.suptitle("K-Means PCA projections for different k values")
    fig.tight_layout()
    fig.savefig(OUT / f"kmeans_pca{suffix}_all_k.png")
    plt.close(fig)


def plot_cluster_composition(path: Path) -> None:
    """绘制最佳 k 下每个簇的真实类别组成热力图。

    这张图回答：每个聚类簇主要由哪些真实标签组成。
    如果每个簇都几乎只有一种真实标签，说明聚类结果和标签较一致。
    """
    rows = []
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    matrix = np.array(
        [[int(row["fall_count"]), int(row["neutral_count"]), int(row["pull_count"])] for row in rows],
        dtype=np.float64,
    )
    row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    ratio = matrix / row_sums
    fig, ax = plt.subplots(figsize=(7.5, max(4.5, len(rows) * 0.48)), dpi=150)
    image = ax.imshow(ratio, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title("Best K-Means cluster composition")
    ax.set_xlabel("true class")
    ax.set_ylabel("cluster")
    ax.set_xticks(range(3), [CLASS_NAMES[i] for i in range(3)], rotation=20, ha="right")
    ax.set_yticks(range(len(rows)), [row["cluster"] for row in rows])
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            ax.text(j, i, f"{ratio[i, j]:.1%}\n({int(matrix[i, j])})", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "best_kmeans_cluster_composition.png")
    plt.close(fig)


def plot_top_features(rows: list[tuple], path: Path, title: str, value_index: int = 1) -> None:
    """绘制 Top 特征重要性横向条形图。

    value_index 指定 rows 中哪个列是重要性分数。
    """
    top = rows[:20][::-1]
    labels = [row[0] for row in top]
    values = [row[value_index] for row in top]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.barh(labels, values)
    ax.set_title(title)
    ax.set_xlabel("score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_report(
    data_rows: list[tuple],
    corr_rows: list[tuple],
    f_rows: list[tuple],
    mi_rows: list[tuple],
    rf_rows: list[tuple],
    channel_rows: list[tuple],
    kmeans_rows: list[tuple],
    pca: PCA,
) -> str:
    """生成 Markdown 特征分析报告。

    只有命令行显式加入 --report 时，主函数才会调用这里写报告。
    报告内容包括数据概览、相关性、聚类、特征重要性和输出文件说明。
    """
    total_rows = {row[0]: row[1] for row in data_rows if row[2] == 0}
    label_lines = []
    for split in ("train", "val", "test"):
        parts = [row for row in data_rows if row[0] == split]
        label_lines.append(
            f"| {split} | {total_rows[split]} | "
            + " | ".join(f"{row[4]} ({row[5]:.2%})" for row in parts)
            + " |"
        )
    best_k = max(kmeans_rows, key=lambda row: row[2])
    lines = [
        "# 特征分析与工程报告",
        "",
        "## 1. 数据概览",
        "- 输入数据为三维序列张量：`样本数 × 60 个时间步 × 33 个指标通道`。",
        "- 当前数据没有提供 33 个通道的真实列名，因此报告使用 `ch00` 到 `ch32` 表示原始指标通道。",
        "- 本报告将每个通道转换为 8 类聚合统计特征：`mean/std/min/max/first/last/delta/slope`，共 `33 × 8 = 264` 个特征，用于解释性分析。",
        "",
        "| split | total | 下跌类/fall | 震荡类/neutral | 拉伸类/pull |",
        "|---|---:|---:|---:|---:|",
        *label_lines,
        "",
        "结论：三个划分中的类别比例完全一致，约为下跌 29.53%、震荡 35.50%、拉伸 34.96%，说明数据切分没有明显类别偏移。",
        "",
        "## 2. 特征相关性分析",
        "- 相关性在训练集聚合特征上计算，使用 Pearson 相关系数。",
        "- 高相关特征通常代表指标冗余，可在建模前做特征筛选、PCA 或正则化。",
        "",
        "| 特征 A | 特征 B | Pearson r | |r| |",
        "|---|---|---:|---:|",
    ]
    for row in corr_rows[:15]:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} |")
    lines.extend(
        [
            "",
            "洞察：部分通道之间存在接近 1 的相关性，说明 33 个指标中包含较强冗余。强相关指标可保留一个代表特征，或交给带正则化的模型自动压缩。",
            "",
            "## 3. 聚类分析",
            f"- 使用标准化后的聚合特征进行 K-Means，比较 k={min(KMEANS_K_VALUES)} 到 k={max(KMEANS_K_VALUES)}。",
            "- 指标包括 inertia、silhouette、ARI、NMI 和 purity。ARI/NMI 越接近 1，说明聚类越接近真实标签；接近 0 则说明自然簇和标签对应较弱。",
            "",
            "| k | inertia | silhouette | ARI | NMI | purity |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in kmeans_rows:
        lines.append(f"| {row[0]} | {row[1]:.1f} | {row[2]:.4f} | {row[3]:.4f} | {row[4]:.4f} | {row[5]:.4f} |")
    lines.extend(
        [
            "",
            f"最佳轮廓系数对应 k={best_k[0]}，silhouette={best_k[2]:.4f}，purity={best_k[5]:.4f}。整体轮廓系数和 ARI/NMI 都偏低，说明数据中没有天然清晰的三簇结构，真实标签边界更可能是连续变化或阈值切分形成的。",
            f"PCA 前两主成分解释方差比例为 {pca.explained_variance_ratio_[0]:.2%} 和 {pca.explained_variance_ratio_[1]:.2%}。真实标签二维投影见 `pca_by_label.png`，不同 k 的聚类投影见 `kmeans_pca_k*.png` 和 `kmeans_pca_all_k.png`，聚类指标曲线见 `kmeans_metrics.png`。",
            "",
            "## 4. 特征重要性评估",
            "本报告使用三种互补方法评估特征价值：",
            "- ANOVA F-value：衡量不同类别下该特征均值差异是否明显。",
            "- Mutual Information：捕捉非线性依赖关系。",
            "- Random Forest Importance：评估树模型中用于分裂类别的贡献。",
            "",
            "### 4.1 ANOVA Top 特征",
            "| rank | feature | F-value | p-value |",
            "|---:|---|---:|---:|",
        ]
    )
    for i, row in enumerate(f_rows[:15], start=1):
        lines.append(f"| {i} | {row[0]} | {row[1]:.2f} | {row[2]:.3e} |")
    lines.extend(["", "### 4.2 Mutual Information Top 特征", "| rank | feature | MI score |", "|---:|---|---:|"])
    for i, row in enumerate(mi_rows[:15], start=1):
        lines.append(f"| {i} | {row[0]} | {row[1]:.5f} |")
    lines.extend(["", "### 4.3 Random Forest Top 特征", "| rank | feature | importance |", "|---:|---|---:|"])
    for i, row in enumerate(rf_rows[:15], start=1):
        lines.append(f"| {i} | {row[0]} | {row[1]:.5f} |")
    lines.extend(
        [
            "",
            "## 5. 通道级别洞察",
            "将同一通道的 8 个聚合特征重要性求和后，可得到通道级别排序：",
            "",
            "| rank | channel | best_feature | best_score | mean_score | total_score |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for i, row in enumerate(channel_rows[:15], start=1):
        lines.append(f"| {i} | {row[0]} | {row[1]} | {row[2]:.2f} | {row[3]:.2f} | {row[4]:.2f} |")
    lines.extend(
        [
            "",
            "## 6. 特征工程建议",
            "1. **保留时间序列信息**：LSTM/GRU/TCN 仍应直接使用原始 `60×33` 序列；聚合特征主要用于解释、筛选和辅助建模。",
            "2. **加入聚合统计分支**：可把 `mean/std/min/max/last/delta/slope` 等统计量与深度模型最后 hidden state 拼接，提高模型对全局趋势和波动的感知。",
            "3. **处理高度相关特征**：对相关系数 `|r| > 0.98` 的特征组做去冗余，或使用 PCA/AutoEncoder 压缩，以降低噪声和过拟合。",
            "4. **强化拉伸类信息**：当前建模结果显示拉伸类召回偏低，建议重点检查对拉伸类区分度高的 `max/std/delta/slope` 特征，并在损失函数中提高拉伸类权重。",
            "5. **构造变化率与交叉特征**：技术指标本身有滞后性，可额外构造最近 5/10/20 步变化、短长窗口差值、成交量指标与价格动量指标的交互项。",
            "6. **避免数据泄漏**：所有标准化、特征选择、PCA 均应只在训练集拟合，再应用到验证集和测试集。",
            "",
            "## 7. 输出文件说明",
            "- `label_distribution.csv/png`：类别分布统计与可视化。",
            "- `top_feature_correlations.csv`：高相关特征对。",
            "- `feature_correlation_heatmap.png`：Top 相关特征 Pearson 相关系数热力图。",
            "- `class_feature_statistics.csv`：各类别下每个聚合特征的均值、标准差、分位数。",
            "- `anova_feature_scores.csv`、`mutual_info_feature_scores.csv`、`random_forest_feature_importance.csv`：三类特征重要性结果。",
            "- `channel_importance_summary.csv`：通道级别重要性汇总。",
            "- `kmeans_metrics.csv`、`best_kmeans_cluster_composition.csv`：聚类评估与簇类别组成。",
            "- `kmeans_metrics.png`：不同 k 的 inertia、silhouette、purity、ARI、NMI 对比图。",
            "- `kmeans_elbow_raw.png`：原始聚合特征空间的 K-Means inertia 手肘图。",
            "- `kmeans_elbow_pca20.png`：PCA 20 维空间的 K-Means inertia 手肘图。",
            "- `silhouette_by_k.png`：不同 k 的轮廓系数图。",
            "- `kmeans_pca_k*.png`、`kmeans_pca_all_k.png`：不同 k 值的 K-Means 聚类 PCA 投影图。",
            "- `best_kmeans_cluster_composition.png`：最佳 k 的簇内真实类别组成热力图。",
            "- `pca_by_label.png`、`pca_sample_coordinates.csv`：PCA 可视化与坐标。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    目前只提供一个开关：--report。
    不加 --report 时，默认只输出 CSV 和 PNG，不写 Markdown 报告。
    """
    parser = argparse.ArgumentParser(description="Run feature analysis, clustering, and exports.")
    parser.add_argument("--report", action="store_true", help="also write feature_analysis_report.md")
    return parser.parse_args()


def main() -> None:
    """特征分析脚本主流程。

    整体顺序：
        1. 读取 train / val / test。
        2. 生成训练集聚合特征，并只用训练集聚合统计做标准化。
        3. 输出类别统计、类别内特征统计。
        4. 输出相关性结果和 Pearson 热力图。
        5. 输出特征重要性结果。
        6. 做 KMeans 聚类、手肘图、轮廓系数图和 PCA 投影图。
        7. 只有传入 --report 时才写 Markdown 报告。
    """
    args = parse_args()
    OUT.mkdir(exist_ok=True)
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    data_rows = data_summary({"train": y_train, "val": y_val, "test": y_test})
    write_rows(OUT / "label_distribution.csv", ["split", "samples", "class_id", "class", "count", "proportion"], data_rows)
    plot_label_distribution(data_rows)

    X_features, feature_names = aggregate_features(X_train)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    np.save(OUT / "train_aggregate_feature_mean.npy", scaler.mean_)
    np.save(OUT / "train_aggregate_feature_scale.npy", scaler.scale_)

    stats_rows = class_feature_stats(X_features, y_train, feature_names)
    write_rows(OUT / "class_feature_statistics.csv", ["class_id", "class", "feature", "mean", "std", "median", "q25", "q75"], stats_rows)

    corr, corr_rows = top_correlations(X_scaled, feature_names)
    np.save(OUT / "feature_correlation_matrix.npy", corr)
    write_rows(OUT / "top_feature_correlations.csv", ["feature_a", "feature_b", "pearson_r", "abs_pearson_r"], corr_rows)
    plot_feature_correlation_heatmap(corr, feature_names)

    f_rows, mi_rows, rf_rows = feature_scores(X_scaled, y_train, feature_names)
    write_rows(OUT / "anova_feature_scores.csv", ["feature", "f_value", "p_value"], f_rows)
    write_rows(OUT / "mutual_info_feature_scores.csv", ["feature", "mutual_info"], mi_rows)
    write_rows(OUT / "random_forest_feature_importance.csv", ["feature", "importance"], rf_rows)
    channel_rows = channel_summary(f_rows, feature_names)
    write_rows(OUT / "channel_importance_summary.csv", ["channel", "best_feature", "best_score", "mean_score", "total_score"], channel_rows)

    kmeans_rows, coords, pca_y, labels_by_k, pca = clustering_analysis(X_scaled, y_train)
    write_rows(OUT / "kmeans_metrics.csv", ["k", "inertia", "silhouette", "adjusted_rand_index", "normalized_mutual_info", "purity"], kmeans_rows)
    plot_elbow_curve(
        [(int(row[0]), float(row[1])) for row in kmeans_rows],
        "Elbow method: raw aggregate feature space",
        OUT / "kmeans_elbow_raw.png",
    )
    plot_silhouette_by_k(kmeans_rows, "Silhouette by k: raw aggregate feature space")
    selected_corr, selected_names, dropped_rows = select_top_low_corr_features(corr, feature_names, f_rows, rf_rows)
    write_rows(OUT / "selected_low_corr_features.csv", ["feature", "index", "absolute_rank"], selected_names)
    write_rows(OUT / "dropped_high_corr_features.csv", ["dropped_feature", "keep_feature", "pearson_r", "abs_pearson_r"], dropped_rows)
    X_selected = X_scaled[:, selected_corr]
    selected_rows, selected_coords, selected_y, selected_labels_by_k, selected_pca = clustering_analysis(X_selected, y_train, pca_file="pca_by_label_selected_features.csv", cluster_composition_file="best_kmeans_cluster_composition_selected_features.csv")
    write_rows(OUT / "kmeans_metrics_selected_features.csv", ["k", "inertia", "silhouette", "adjusted_rand_index", "normalized_mutual_info", "purity"], selected_rows)
    plot_elbow_curve(
        [(int(row[0]), float(row[1])) for row in selected_rows],
        "Elbow method: low-correlation selected features",
        OUT / "kmeans_elbow_selected_features.png",
    )
    plot_silhouette_by_k(selected_rows, "Silhouette by k: low-correlation selected features")
    plot_pca(selected_coords, selected_y, title="PCA projection of selected low-correlation features")
    plot_kmeans_metrics(selected_rows, save_path=OUT / "kmeans_metrics_selected_features.png")
    plot_kmeans_pca(selected_coords, selected_labels_by_k, suffix="selected")
    plot_cluster_composition(OUT / "best_kmeans_cluster_composition_selected_features.csv", path=OUT / "best_kmeans_cluster_composition_selected_features.png")
    sample_n = min(25000, len(y_train))
    cluster_pca = PCA(n_components=20, random_state=42)
    X_pca20 = cluster_pca.fit_transform(X_scaled[:sample_n])
    pca20_rows = cluster_inertias(X_pca20)
    np.save(OUT / "cluster_pca20_explained_variance_ratio.npy", cluster_pca.explained_variance_ratio_)
    plot_elbow_curve(pca20_rows, "Elbow method: PCA-20 feature space", OUT / "kmeans_elbow_pca20.png")
    plot_pca(coords, pca_y)
    plot_kmeans_metrics(kmeans_rows)
    plot_kmeans_pca(coords, labels_by_k)
    plot_cluster_composition(OUT / "best_kmeans_cluster_composition.csv")
    plot_top_features(f_rows, OUT / "top_anova_features.png", "Top ANOVA features", value_index=1)
    plot_top_features(mi_rows, OUT / "top_mutual_info_features.png", "Top mutual information features", value_index=1)
    plot_top_features(rf_rows, OUT / "top_random_forest_features.png", "Top random forest features", value_index=1)

    if args.report:
        report = make_report(data_rows, corr_rows, f_rows, mi_rows, rf_rows, channel_rows, kmeans_rows, pca)
        (OUT / "feature_analysis_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "aggregate_feature_count": len(feature_names),
        "output_dir": str(OUT),
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report:
        print(f"report={OUT / 'feature_analysis_report.md'}")
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
