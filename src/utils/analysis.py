# C:\Temp\Pcharm\floating_wind_platform\src\utils\analysis.py (全新文件)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def analyze_dynamic_weights(coeffs: np.ndarray, x_hist: np.ndarray, feature_names: list, save_prefix: str):
    """
    深度分析并可视化动态权重 (增强版)
    """
    logger.info("开始进行深度动态权重分析...")

    # 1. 权重统计分析 (增强)
    w0, w1, w2 = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2]

    report = "动态权重分析报告\n" + "=" * 50 + "\n"

    for i, w in enumerate([w0, w1, w2]):
        name = ["偏置项 (w0)", "PatchTST 权重 (w1)", "NHITS 权重 (w2)"][i]
        report += f"{name} 统计:\n"
        report += f"  均值: {np.mean(w):.4f}, 标准差: {np.std(w):.4f}\n"
        report += f"  最小值: {np.min(w):.4f}, 最大值: {np.max(w):.4f}\n"
        report += f"  25% 分位数: {np.percentile(w, 25):.4f}\n"
        report += f"  50% 分位数 (中位数): {np.percentile(w, 50):.4f}\n"
        report += f"  75% 分位数: {np.percentile(w, 75):.4f}\n\n"

    # 2. 权重间相关性 (增强)
    weight_correlation = np.corrcoef(w1.flatten(), w2.flatten())[0, 1]
    report += f"PatchTST (w1) 与 NHITS (w2) 权重相关性: {weight_correlation:.4f}\n"
    report += "一个显著的负相关是期望的结果，表明专家模型形成了有效的分工与互补。\n\n"

    with open(f"{save_prefix}_分析报告.txt", "w", encoding='utf-8') as f:
        f.write(report)
    logger.info(f"权重分析报告已保存至: {save_prefix}_分析报告.txt")

    # 3. 可视化：权重分布 (增强)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.histplot(w0.flatten(), kde=True, ax=axes[0]).set_title('偏置项 (w0) 分布', fontsize=16)
    sns.histplot(w1.flatten(), kde=True, ax=axes[1]).set_title('PatchTST 权重 (w1) 分布', fontsize=16)
    sns.histplot(w2.flatten(), kde=True, ax=axes[2]).set_title('NHITS 权重 (w2) 分布', fontsize=16)
    fig.suptitle('动态融合系数分布图', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_prefix}_权重分布.png");
    plt.close()

    # 4. 可视化：特征与权重的相关性 (增强)
    context_features = x_hist[:, -1, :]
    weights_at_step_0 = coeffs[:, 0, 1:]

    corr_df = pd.DataFrame(context_features, columns=feature_names)
    corr_df['PatchTST 权重 (w1)'] = weights_at_step_0[:, 0]
    corr_df['NHITS 权重 (w2)'] = weights_at_step_0[:, 1]

    feature_weight_corr = corr_df.corr().loc[feature_names, ['PatchTST 权重 (w1)', 'NHITS 权重 (w2)']]

    plt.figure(figsize=(10, 8))
    sns.heatmap(feature_weight_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("输入特征与门控权重相关性\n(基于历史序列最后一个时间步)", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_特征与权重相关性热力图.png");
    plt.close()

    logger.info("权重分析图表已生成。")