"""
真实数据验证脚本
基于浮式风机平台.xlsx数据进行快速验证
"""

import numpy as np
import torch
import sys
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_preprocessing.data_loader import DataLoader
from models.patchtst import PatchTST
from models.nhits import NHITS
from src.strategies.mpa_optimizer import MPAOptimizer, StackingOptimizer

from src.evaluation.metrics import EvaluationMetrics
from src.visualization.plots import VisualizationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data_loading():
    """验证数据加载"""
    print("=" * 60)
    print("步骤1: 验证真实数据加载")
    print("=" * 60)
    
    try:
        # 使用配置文件加载真实数据
        config_path = project_root / "configs" / "config.yaml"
        data_loader = DataLoader(str(config_path))
        
        # 加载数据
        data = data_loader.load_data()
        print(f"✓ 数据加载成功: {data.shape}")
        print(f"✓ 数据列: {list(data.columns)}")
        
        # 数据预处理
        target_scaled, covariate_scaled, time_stamps = data_loader.preprocess_data()
        print(f"✓ 目标变量标准化: {target_scaled.shape}")
        print(f"✓ 协变量标准化: {covariate_scaled.shape}")
        
        # 创建序列
        X, y = data_loader.create_sequences(target_scaled, covariate_scaled)
        print(f"✓ 序列创建完成: X{X.shape}, y{y.shape}")
        
        # 数据划分
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
        print(f"✓ 训练集: X{X_train.shape}, y{y_train.shape}")
        print(f"✓ 验证集: X{X_val.shape}, y{y_val.shape}")
        print(f"✓ 测试集: X{X_test.shape}, y{y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"✗ 数据加载失败: {str(e)}")
        raise


def validate_models_quick(X_train, y_train, X_val, y_val):
    """快速验证模型（使用少量epoch）"""
    """快速验证模型（使用少量epoch）"""
    # 添加输入验证检查
    if X_train is None or y_train is None or X_val is None or y_val is None:
        raise ValueError("validate_models_quick: 输入数据不能为None。请先加载数据。")

    if len(X_train) == 0 or len(y_train) == 0 or len(X_val) == 0 or len(y_val) == 0:
        raise ValueError("validate_models_quick: 输入数据不能为空数组。")

    if X_train.shape[0] != y_train.shape[0] or X_val.shape[0] != y_val.shape[0]:
        raise ValueError("validate_models_quick: 输入数据的样本数量不匹配。")

    print("\n" + "=" * 60)
    print("步骤2: 快速模型验证（5个epoch）")
    print("=" * 60)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数
    batch_size = 32
    input_size = X_train.shape[1]
    horizon = y_train.shape[1]
    num_features = X_train.shape[2]
    
    print(f"模型参数: input_size={input_size}, horizon={horizon}, num_features={num_features}")
    
    # 训练PatchTST（快速
    #
    # 模式）
    print("\n训练PatchTST模型...")
    patchtst = PatchTST(input_size=input_size, horizon=horizon, num_features=num_features)
    patchtst_train_loss = train_model_quick(patchtst, X_train, y_train, device, epochs=5)
    
    # 训练NHITS（快速模式）
    print("\n训练NHITS模型...")
    nhits = NHITS(input_size=input_size, horizon=horizon, num_features=num_features)
    nhits_train_loss = train_model_quick(nhits, X_train, y_train, device, epochs=5)
    
    # 验证集预测
    print("\n验证集预测...")
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        patchtst_pred = patchtst(X_val_tensor).cpu().numpy()
        nhits_pred = nhits(X_val_tensor).cpu().numpy()
    
    # 计算验证指标
    patchtst_metrics = EvaluationMetrics.calculate_all_metrics(y_val, patchtst_pred)
    nhits_metrics = EvaluationMetrics.calculate_all_metrics(y_val, nhits_pred)
    
    print(f"\nPatchTST验证指标:")
    print(f"  RMSE: {patchtst_metrics['RMSE']:.6f}")
    print(f"  MAE: {patchtst_metrics['MAE']:.6f}")
    print(f"  MAPE: {patchtst_metrics['MAPE']:.6f}%")
    
    print(f"\nNHITS验证指标:")
    print(f"  RMSE: {nhits_metrics['RMSE']:.6f}")
    print(f"  MAE: {nhits_metrics['MAE']:.6f}")
    print(f"  MAPE: {nhits_metrics['MAPE']:.6f}%")
    
    return patchtst, nhits, patchtst_pred, nhits_pred, patchtst_metrics, nhits_metrics


def train_model_quick(model, X_train, y_train, device, epochs=5):
    """快速训练模型"""
    # 添加输入验证检查
    if model is None:
        raise ValueError("train_model_quick: 模型不能为None。")

    if X_train is None or y_train is None:
        raise ValueError("train_model_quick: 训练数据不能为None。")

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("train_model_quick: 训练数据不能为空数组。")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("train_model_quick: X_train和y_train的样本数量不匹配。")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # 简单的数据加载器
    batch_size = 32
    n_batches = len(X_train) // batch_size
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            
            X_batch = torch.FloatTensor(X_train[start_idx:end_idx]).to(device)
            y_batch = torch.FloatTensor(y_train[start_idx:end_idx]).to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        train_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return train_losses


def validate_fusion_strategies(y_true, patchtst_pred, nhits_pred):
    """验证融合策略（使用真实数据）"""
    # 添加输入验证检查
    if y_true is None or patchtst_pred is None or nhits_pred is None:
        raise ValueError("validate_fusion_strategies: 输入预测数据不能为None。")

    if len(y_true) == 0 or len(patchtst_pred) == 0 or len(nhits_pred) == 0:
        raise ValueError("validate_fusion_strategies: 输入预测数据不能为空数组。")

    if y_true.shape != patchtst_pred.shape or y_true.shape != nhits_pred.shape:
        raise ValueError("validate_fusion_strategies: 输入预测数据的形状不匹配。")

    print("\n" + "=" * 60)
    print("步骤3: 验证融合策略（MPA 20次迭代）")
    print("=" * 60)
    
    # 准备专家预测数据
    expert_predictions = np.stack([patchtst_pred, nhits_pred], axis=2)
    print(f"专家预测数据形状: {expert_predictions.shape}")
    print(f"真实值形状: {y_true.shape}")
    
    # MPA配置（快速模式）
    mpa_config = {
        'population_size': 20,
        'max_iterations': 20,  # 快速验证
        'fads_probability': 0.2,
        'convergence_threshold': 1e-4
    }
    
    # 策略A：静态权重优化
    print("\n策略A: 静态权重优化...")
    static_optimizer = MPAOptimizer.StaticWeightOptimizer(mpa_config)
    weights_a, score_a = static_optimizer.optimize_weights(expert_predictions, y_true)
    
    # 计算策略A预测
    strategy_a_pred = np.zeros_like(y_true)
    for i, weight in enumerate(weights_a):
        strategy_a_pred += weight * expert_predictions[:, :, i]
    
    print(f"最优权重: {weights_a}")
    print(f"RMSE: {score_a:.6f}")
    
    # 策略B：广义线性融合
    print("\n策略B: 广义线性融合...")
    stacking_optimizer = StackingOptimizer(mpa_config)
    coefficients_b, score_b = stacking_optimizer.optimize_coefficients(expert_predictions, y_true)
    
    # 计算策略B预测
    w0 = coefficients_b[0]
    weights_b = coefficients_b[1:]
    strategy_b_pred = np.full_like(y_true, w0)
    for i, weight in enumerate(weights_b):
        strategy_b_pred += weight * expert_predictions[:, :, i]
    
    print(f"最优系数: {coefficients_b}")
    print(f"RMSE: {score_b:.6f}")
    
    # 计算各策略指标
    strategy_a_metrics = EvaluationMetrics.calculate_all_metrics(y_true, strategy_a_pred)
    strategy_b_metrics = EvaluationMetrics.calculate_all_metrics(y_true, strategy_b_pred)
    
    print(f"\n策略A指标: RMSE={strategy_a_metrics['RMSE']:.6f}, MAE={strategy_a_metrics['MAE']:.6f}")
    print(f"策略B指标: RMSE={strategy_b_metrics['RMSE']:.6f}, MAE={strategy_b_metrics['MAE']:.6f}")
    
    return {
        'strategy_a': {'pred': strategy_a_pred, 'metrics': strategy_a_metrics, 'weights': weights_a},
        'strategy_b': {'pred': strategy_b_pred, 'metrics': strategy_b_metrics, 'coefficients': coefficients_b},
        'experts': {'patchtst': patchtst_pred, 'nhits': nhits_pred}
    }


def generate_validation_plots(y_true, results):
    """生成验证图表"""
    print("\n" + "=" * 60)
    print("步骤4: 生成验证图表")
    print("=" * 60)
    
    # 创建可视化引擎
    viz_engine = VisualizationEngine("real_validation_results")
    
    # 准备预测数据
    predictions = {
        'PatchTST': results['experts']['patchtst'],
        'NHITS': results['experts']['nhits'],
        'Strategy_A_Static': results['strategy_a']['pred'],
        'Strategy_B_Stacking': results['strategy_b']['pred']
    }
    
    # 性能对比图
    print("生成性能对比图...")
    metrics = {
        'PatchTST': results['experts']['patchtst_metrics'],
        'NHITS': results['experts']['nhits_metrics'],
        'Strategy_A_Static': results['strategy_a']['metrics'],
        'Strategy_B_Stacking': results['strategy_b']['metrics']
    }
    
    perf_path = viz_engine.plot_performance_comparison(metrics)
    print(f"✓ 性能对比图: {perf_path}")
    
    # 时间序列对比图（前3个样本）
    print("生成时间序列对比图...")
    for i in range(min(3, len(y_true))):
        ts_path = viz_engine.plot_time_series_comparison(y_true, predictions, sample_idx=i)
        print(f"✓ 样本{i}时间序列图: {ts_path}")
    
    # 峰值误差分析
    print("生成峰值误差分析图...")
    peak_path = viz_engine.plot_peak_error_analysis(y_true, predictions)
    print(f"✓ 峰值误差分析: {peak_path}")
    
    # 残差分析
    print("生成残差分析图...")
    residual_path = viz_engine.plot_residual_analysis(y_true, predictions)
    print(f"✓ 残差分析: {residual_path}")
    
    return viz_engine


def main():
    """主函数：真实数据快速验证"""
    print("🌊 浮式风机平台 - 真实数据快速验证")
    print("=" * 60)
    print("使用真实数据验证系统功能（快速模式）")
    print("预期运行时间: 15-20分钟")
    print("=" * 60)
    
    try:
        # 步骤1: 数据加载验证
        X_train, X_val, X_test, y_train, y_val, y_test = validate_data_loading()
        
        # 步骤2: 模型快速验证
        patchtst, nhits, patchtst_pred, nhits_pred, patchtst_metrics, nhits_metrics = \
            validate_models_quick(X_train, y_train, X_val, y_val)
        
        # 存储专家指标供后续使用
        experts_data = {
            'patchtst': {'pred': patchtst_pred, 'metrics': patchtst_metrics},
            'nhits': {'pred': nhits_pred, 'metrics': nhits_metrics}
        }
        
        # 步骤3: 融合策略验证
        results = validate_fusion_strategies(y_val, patchtst_pred, nhits_pred)
        results['experts'] = experts_data
        
        # 步骤4: 生成图表
        viz_engine = generate_validation_plots(y_val, results)
        
        # 总结
        print("\n" + "=" * 60)
        print("✅ 真实数据验证完成！")
        print("=" * 60)
        print("验证结果摘要:")
        print(f"数据加载: ✓ 成功 ({X_train.shape[0]}训练样本)")
        print(f"PatchTST: RMSE={patchtst_metrics['RMSE']:.6f}")
        print(f"NHITS: RMSE={nhits_metrics['RMSE']:.6f}")
        print(f"策略A: RMSE={results['strategy_a']['metrics']['RMSE']:.6f}, 权重={results['strategy_a']['weights']}")
        print(f"策略B: RMSE={results['strategy_b']['metrics']['RMSE']:.6f}")
        print("\n图表已保存到 results/real_validation_results/ 目录")
        print("✅ 系统验证通过，可以进行完整实验！")
        
    except Exception as e:
        print(f"\n❌ 验证失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()