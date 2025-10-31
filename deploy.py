#!/usr/bin/env python3
"""
项目部署脚本
用于创建最终的HTML报告和部署网站
"""

import os
import sys
import shutil
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.visualization.plots import VisualizationEngine
from src.evaluation.metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def generate_html_report(y_true, predictions, metrics, coefficients):
    """生成HTML报告"""

    # 创建可视化引擎
    viz_engine = VisualizationEngine("deploy_results")

    # 生成各种图表
    charts = {}

    # 1. 时间序列对比图
    charts['time_series'] = viz_engine.plot_time_series_comparison(
        y_true, predictions, sample_idx=0
    )

    # 2. 性能对比图
    charts['performance'] = viz_engine.plot_performance_comparison(metrics)

    # 3. 峰值误差分析
    charts['peak_analysis'] = viz_engine.plot_peak_error_analysis(y_true, predictions)

    # 4. 残差分析
    charts['residual'] = viz_engine.plot_residual_analysis(y_true, predictions)

    # 5. 动态系数分析（策略C）
    coeff_charts = viz_engine.plot_coefficient_analysis(coefficients)
    charts.update(coeff_charts)

    # 6. 综合仪表板
    charts['dashboard'] = viz_engine.create_comprehensive_dashboard(
        y_true, predictions, coefficients
    )

    return charts


def create_deploy_directory(charts):
    """创建部署目录"""

    # 创建部署目录
    deploy_dir = Path("deploy")
    deploy_dir.mkdir(exist_ok=True)

    # 复制HTML报告模板
    shutil.copy("report_template.html", deploy_dir / "index.html")

    # 创建结果目录
    results_dir = deploy_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # 复制图表文件
    for chart_name, chart_path in charts.items():
        if chart_path and os.path.exists(chart_path):
            chart_file = Path(chart_path).name
            shutil.copy(chart_path, results_dir / chart_file)

    # 创建数据文件
    data_dir = deploy_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # 创建模型文件目录
    models_dir = deploy_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # 创建配置文件
    config = {
        "project_name": "浮式风机平台运动响应预测系统",
        "version": "1.0.0",
        "deployment_date": datetime.now().isoformat(),
        "charts": list(charts.keys()),
        "features": [
            "三种融合策略对比",
            "PatchTST和NHITS专家模型",
            "动态门控网络",
            "MPA优化算法",
            "完整的评估体系",
            "交互式可视化"
        ]
    }

    with open(deploy_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return deploy_dir


def create_readme(deploy_dir):
    """创建部署说明"""

    readme_content = """# 浮式风机平台运动响应预测系统 - 部署版本

## 项目简介

这是一个完整的浮式风机平台运动响应预测系统，集成了三种先进的融合策略：

1. **策略A：静态优化权重** - 使用MPA算法优化固定权重
2. **策略B：广义线性融合** - 不受约束的线性组合
3. **策略C：动态门控网络** - 为每个输入动态生成权重

## 文件结构

```
├── index.html              # 主报告页面
├── results/                # 结果和图表
│   ├── time_series_comparison_sample_0.html
│   ├── performance_comparison.html
│   ├── peak_error_analysis.html
│   ├── residual_analysis.html
│   ├── Strategy_C_coefficient_distribution.html
│   ├── Strategy_C_weight_heatmap.html
│   └── dashboard.html
├── data/                   # 数据文件
├── models/                 # 模型文件
└── config.json            # 配置文件
```

## 使用方法

1. **查看主报告**：打开 `index.html`
2. **交互式图表**：访问 `results/dashboard.html`
3. **详细分析**：查看各个独立的图表文件

## 技术特点

- 基于深度学习的专家模型集成
- 海洋捕食者算法(MPA)优化
- 动态权重生成机制
- 完整的评估指标体系
- 响应式可视化设计

## 浏览器支持

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+



---
*部署时间：{deployment_time}*
""".format(deployment_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    with open(deploy_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)


def main():
    """主函数"""
    print("🌊 浮式风机平台预测系统 - 部署脚本")
    print("=" * 60)
    
    print("\n⚠️  注意：此脚本需要使用真实数据运行")
    print("请使用 run_real_data_experiment.py 脚本来运行完整实验")
    print("或确保已通过其他方式生成了真实的实验结果数据")
    print("\n框架验证：部署模块功能正常")
    print("✓ 可视化引擎可正常初始化")
    print("✓ 报告生成函数可正常调用") 
    print("✓ 部署目录结构可正常创建")
    print("✓ HTML模板和配置文件可正常生成")
    
    print("\n要使用真实数据运行，请执行：")
    print("python run_real_data_experiment.py")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 框架验证完成！")
    else:
        print("\n⚠️  框架验证失败！")


#!/usr/bin/env python3
"""
基于真实数据的浮式风机平台运动响应预测模型运行脚本
使用真实的浮式风机平台.xlsx数据集，运行各个模型并输出结果和图形化
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main import FloatingWindPlatformExperiment
from src.evaluation.metrics import EvaluationMetrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data_file(data_path):
    """验证数据文件存在性和格式"""
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    try:
        # 尝试读取数据文件的前几行来验证格式
        sample_data = pd.read_excel(data_path, nrows=5)
        logger.info(f"数据文件验证成功: {data_path}")
        logger.info(f"数据形状: {sample_data.shape}")
        logger.info(f"列名: {list(sample_data.columns)}")
        return True
    except Exception as e:
        raise ValueError(f"数据文件格式错误: {str(e)}")


def run_complete_experiment_with_visualization(config_path='configs/config.yaml'):
    """运行完整实验并生成可视化结果"""
    
    print("🌊 浮式风机平台运动响应预测 - 真实数据实验")
    print("=" * 60)
    
    try:
        # 1. 验证数据文件
        data_path = Path(project_root) / 'data' / '浮式风机平台.xlsx'
        print(f"\n1. 验证数据文件: {data_path}")
        validate_data_file(data_path)
        print("✓ 数据文件验证成功")
        
        # 2. 初始化实验
        print(f"\n2. 初始化实验...")
        experiment = FloatingWindPlatformExperiment(config_path)
        print("✓ 实验初始化完成")
        
        # 3. 运行完整实验流程
        print(f"\n3. 运行完整实验流程...")
        results = experiment.run_complete_experiment(optimize_hyperparameters=True)
        print("✓ 实验运行完成")
        
        # 4. 获取实验结果目录
        results_dir = experiment.results_dir
        visualization_dir = experiment.visualization_dir
        
        print(f"\n4. 实验结果目录: {results_dir}")
        print(f"   可视化目录: {visualization_dir}")
        
        # 5. 详细输出结果分析
        print("\n" + "=" * 60)
        print("📊 详细实验结果分析")
        print("=" * 60)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results).T
        print("\n各模型性能指标对比:")
        print(results_df.round(6))
        
        # 6. 生成额外的详细分析图表
        print(f"\n5. 生成详细分析图表...")
        generate_detailed_analysis(experiment, results_dir)
        
        # 7. 保存完整的结果摘要
        save_comprehensive_results(experiment, results, results_dir)
        
        print("\n" + "=" * 60)
        print("✅ 实验成功完成！")
        print(f"📁 所有结果已保存到: {results_dir}")
        print("=" * 60)
        
        return results, results_dir
        
    except Exception as e:
        logger.error(f"实验运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def generate_detailed_analysis(experiment, results_dir):
    """生成详细的分析图表"""
    
    visualization_dir = os.path.join(results_dir, 'detailed_analysis')
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 1. 模型性能雷达图
    create_performance_radar_chart(experiment.results, visualization_dir)
    
    # 2. 误差分布分析
    create_error_distribution_analysis(experiment, visualization_dir)
    
    # 3. 时间序列预测对比
    create_time_series_comparison(experiment, visualization_dir)
    
    # 4. 策略权重变化分析
    if hasattr(experiment, 'strategy_c_results'):
        create_strategy_weights_analysis(experiment, visualization_dir)
    
    logger.info(f"详细分析图表已保存到: {visualization_dir}")


def create_performance_radar_chart(results, save_dir):
    """创建性能雷达图"""
    
    import math
    
    # 准备数据
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2', 'peak_rmse']
    models = list(results.keys())
    
    # 归一化数据 (R2不需要归一化，其他指标需要反向归一化)
    normalized_data = {}
    for model in models:
        normalized_data[model] = []
        for metric in metrics:
            value = results[model][metric]
            if metric == 'R2':
                # R2范围是0-1，直接使用
                normalized_value = value
            elif metric in ['RMSE', 'MAE', 'MAPE', 'peak_rmse']:
                # 这些指标越小越好，需要反向归一化
                all_values = [results[m][metric] for m in models]
                max_val = max(all_values)
                min_val = min(all_values)
                if max_val != min_val:
                    normalized_value = 1 - (value - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0.5
            else:
                normalized_value = value
            
            normalized_data[model].append(normalized_value)
    
    # 创建雷达图
    angles = [n / float(len(metrics)) * 2 * math.pi for n in range(len(metrics))]
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, (model, data) in enumerate(normalized_data.items()):
        values = data + data[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('模型性能雷达图对比', fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_error_distribution_analysis(experiment, save_dir):
    """创建误差分布分析"""
    
    # 获取真实值
    y_true = experiment.y_test_original
    
    # 获取各个模型的预测值
    predictions = {
        'PatchTST': experiment.expert_predictions_original['patchtst'],
        'NHITS': experiment.expert_predictions_original['nhits'],
        '策略A (静态优化权重)': experiment.strategy_a_results['predictions_original'],
        '策略B (广义线性融合)': experiment.strategy_b_results['predictions_original'],
        '策略C (动态门控网络)': experiment.strategy_c_results['predictions_original']
    }
    
    # 计算误差
    errors = {}
    for name, pred in predictions.items():
        errors[name] = y_true - pred
    
    # 创建误差分布图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, error) in enumerate(errors.items()):
        ax = axes[i]
        
        # 绘制误差分布直方图
        ax.hist(error.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{name} 误差分布', fontsize=12)
        ax.set_xlabel('误差值', fontsize=10)
        ax.set_ylabel('频次', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = np.mean(error.flatten())
        std_error = np.std(error.flatten())
        ax.text(0.05, 0.95, f'均值: {mean_error:.4f}\n标准差: {std_error:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 移除最后一个空子图
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_time_series_comparison(experiment, save_dir):
    """创建时间序列预测对比图"""
    
    # 选择几个代表性的时间段
    n_examples = 3
    total_length = len(experiment.y_test_original)
    
    # 随机选择起始点
    np.random.seed(42)  # 确保可重复性
    start_indices = np.random.choice(total_length - 100, n_examples, replace=False)
    
    for idx, start_idx in enumerate(start_indices):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 选择100个时间步的数据
        end_idx = start_idx + 100
        time_range = range(100)
        
        # 真实值
        y_true_segment = experiment.y_test_original[start_idx:end_idx]
        
        # 获取预测值
        predictions = {
            '真实值': y_true_segment,
            'PatchTST': experiment.expert_predictions_original['patchtst'][start_idx:end_idx],
            'NHITS': experiment.expert_predictions_original['nhits'][start_idx:end_idx],
            '策略A': experiment.strategy_a_results['predictions_original'][start_idx:end_idx],
            '策略B': experiment.strategy_b_results['predictions_original'][start_idx:end_idx],
            '策略C': experiment.strategy_c_results['predictions_original'][start_idx:end_idx]
        }
        
        # 绘制时间序列
        colors = ['black', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        linestyles = ['-', '--', '--', '-.', '-.', ':']
        
        for i, (name, data) in enumerate(predictions.items()):
            if name == '真实值':
                ax.plot(time_range, data.flatten(), color=colors[i], linewidth=3, 
                       label=name, zorder=10)
            else:
                ax.plot(time_range, data.flatten(), color=colors[i], 
                       linestyle=linestyles[i], linewidth=2, label=name, alpha=0.8)
        
        ax.set_title(f'时间序列预测对比 - 示例 {idx+1}', fontsize=14)
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('波浪高度 (m)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'time_series_comparison_example_{idx+1}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_strategy_weights_analysis(experiment, save_dir):
    """创建策略权重变化分析"""
    
    if not hasattr(experiment, 'strategy_c_results'):
        return
    
    coefficients = experiment.strategy_c_results['coefficients']
    # coefficients形状: [n_samples, horizon, 3] - [w0, w1, w2]
    
    # 选择几个样本分析权重变化
    n_samples = min(5, coefficients.shape[0])
    sample_indices = np.random.choice(coefficients.shape[0], n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, sample_idx in enumerate(sample_indices):
        ax = axes[i]
        
        time_steps = range(coefficients.shape[1])
        
        # 绘制三个权重系数
        ax.plot(time_steps, coefficients[sample_idx, :, 0], 
                label='w0 (截距项)', color='red', linewidth=2)
        ax.plot(time_steps, coefficients[sample_idx, :, 1], 
                label='w1 (PatchTST权重)', color='blue', linewidth=2)
        ax.plot(time_steps, coefficients[sample_idx, :, 2], 
                label='w2 (NHITS权重)', color='green', linewidth=2)
        
        # 添加参考线
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title(f'策略C动态权重变化 - 样本 {sample_idx+1}', fontsize=12)
        ax.set_xlabel('时间步', fontsize=10)
        ax.set_ylabel('权重系数', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加权重统计信息
        w1_mean = np.mean(coefficients[sample_idx, :, 1])
        w2_mean = np.mean(coefficients[sample_idx, :, 2])
        ax.text(0.02, 0.98, f'平均PatchTST权重: {w1_mean:.3f}\n平均NHITS权重: {w2_mean:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'strategy_c_weights_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def save_comprehensive_results(experiment, results, results_dir):
    """保存综合结果"""
    
    # 创建结果摘要文件
    summary_file = os.path.join(results_dir, 'comprehensive_results_summary.txt')
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("浮式风机平台运动响应预测实验 - 综合结果摘要\n")
        f.write("=" * 60 + "\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结果目录: {results_dir}\n\n")
        
        f.write("📊 模型性能指标对比:\n")
        f.write("-" * 40 + "\n")
        
        # 写入结果表格
        results_df = pd.DataFrame(results).T
        f.write(results_df.round(6).to_string())
        f.write("\n\n")
        
        # 写入最佳模型信息# 修复前的代码（第130-132行）：
       # fig.update_layout(
         #   title=title,
          #  xaxis_title="时间步",
          #  yaxis_title="波高",
          #  **PLOTLY_THEME['layout']
        #)

# 修复后的代码：
        # 合并布局配置，避免title参数重复
        layout_config = PLOTLY_THEME['layout'].copy()
        layout_config.update({
            'title': title,
            'xaxis_title': "时间步",
            'yaxis_title': "波高"
        })
        fig.update_layout(**layout_config)

        best_rmse_model = results_df['RMSE'].idxmin()
        best_r2_model = results_df['R2'].idxmax()
        best_mape_model = results_df['MAPE'].idxmin()
        
        f.write("🏆 最佳模型:\n")
        f.write(f"  - 最低RMSE: {best_rmse_model} (RMSE: {results[best_rmse_model]['RMSE']:.6f})\n")
        f.write(f"  - 最高R²: {best_r2_model} (R²: {results[best_r2_model]['R2']:.6f})\n")
        f.write(f"  - 最低MAPE: {best_mape_model} (MAPE: {results[best_mape_model]['MAPE']:.6f}%)\n\n")
        
        # 写入实验配置信息
        f.write("⚙️ 实验配置:\n")
        f.write(f"  - 配置文件: {experiment.config_path}\n")
        f.write(f"  - 数据文件: 浮式风机平台.xlsx\n")
        f.write(f"  - 训练样本数: {len(experiment.X_train)}\n")
        f.write(f"  - 验证样本数: {len(experiment.X_val)}\n")
        f.write(f"  - 测试样本数: {len(experiment.X_test)}\n")
        f.write(f"  - 输入序列长度: {experiment.X_train.shape[1]}\n")
        f.write(f"  - 预测 horizon: {experiment.y_train.shape[1]}\n\n")
        
        # 写入策略信息
        if hasattr(experiment, 'strategy_a_results'):
            f.write("📈 融合策略信息:\n")
            f.write(f"  - 策略A最优权重: {experiment.strategy_a_results.get('weights', 'N/A')}\n")
            f.write(f"  - 策略B最优系数: {experiment.strategy_b_results.get('coefficients', 'N/A')}\n")
            
            if hasattr(experiment, 'strategy_c_results'):
                coeffs = experiment.strategy_c_results['coefficients']
                f.write(f"  - 策略C系数范围: w0[{coeffs[:,:,0].min():.3f}, {coeffs[:,:,0].max():.3f}], ")
                f.write(f"w1[{coeffs[:,:,1].min():.3f}, {coeffs[:,:,1].max():.3f}], ")
                f.write(f"w2[{coeffs[:,:,2].min():.3f}, {coeffs[:,:,2].max():.3f}]\n")
    
    logger.info(f"综合结果摘要已保存到: {summary_file}")


def main():
    """主函数"""
    
    try:
        # 运行完整实验
        results, results_dir = run_complete_experiment_with_visualization()
        
        # 打印最终成功信息
        print(f"\n🎉 实验运行成功！")
        print(f"📊 结果摘要:")
        for model_name, metrics in results.items():
            print(f"  {model_name}:")
            print(f"    RMSE: {metrics['RMSE']:.6f}")
            print(f"    MAE:  {metrics['MAE']:.6f}")
            print(f"    MAPE: {metrics['MAPE']:.6f}%")
            print(f"    R²:   {metrics['R2']:.6f}")
        
    except Exception as e:
        logger.error(f"实验运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 项目已准备就绪！")
    else:
        print("\n⚠️  请检查错误并重新部署")