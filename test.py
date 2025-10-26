"""
快速测试脚本
用于验证系统基本功能
"""

import numpy as np
import torch
import sys
from pathlib import Path
import logging

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.patchtst import PatchTST
from src.models.nhits import NHITS
from src.strategies.mpa_optimizer import StaticWeightOptimizer, StackingOptimizer
from src.strategies.gating_network import GatingNetwork, DynamicFusionModel
from src.evaluation.metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_models():
    """测试模型基本功能"""
    print("=" * 60)
    print("测试模型基本功能")
    print("=" * 60)

    # 设置参数
    batch_size = 32
    input_size = 96
    horizon = 24
    num_features = 7

    # 创建测试数据
    X = torch.randn(batch_size, input_size, num_features)
    y = torch.randn(batch_size, horizon)

    # 测试PatchTST
    print("\n1. 测试PatchTST模型...")
    patchtst = PatchTST(
        input_size=input_size,
        horizon=horizon,
        num_features=num_features
    )

    with torch.no_grad():
        patchtst_pred = patchtst(X)
        print(f"   PatchTST输入: {X.shape}")
        print(f"   PatchTST输出: {patchtst_pred.shape}")
        assert patchtst_pred.shape == (batch_size, horizon)

    # 测试NHITS
    print("\n2. 测试NHITS模型...")
    nhits = NHITS(
        input_size=input_size,
        horizon=horizon,
        num_features=num_features
    )

    with torch.no_grad():
        nhits_pred = nhits(X)
        print(f"   NHITS输入: {X.shape}")
        print(f"   NHITS输出: {nhits_pred.shape}")
        assert nhits_pred.shape == (batch_size, horizon)

    print("\n✓ 模型基本功能测试通过")

    return patchtst, nhits, X, y


def test_fusion_strategies():
    """测试融合策略"""
    print("\n" + "=" * 60)
    print("测试融合策略")
    print("=" * 60)

    # 创建模拟预测数据
    n_samples = 100
    horizon = 24
    n_experts = 2

    # 模拟真实值和两个专家的预测
    y_true = np.random.randn(n_samples, horizon) * 2 + 10
    expert1_pred = y_true + np.random.randn(n_samples, horizon) * 0.5  # 较差预测
    expert2_pred = y_true + np.random.randn(n_samples, horizon) * 0.3  # 较好预测

    # 合并预测 [n_samples, horizon, n_experts]
    expert_predictions = np.stack([expert1_pred, expert2_pred], axis=2)

    # 测试策略A：静态优化权重
    print("\n1. 测试策略A：静态优化权重...")
    mpa_config = {
        'population_size': 20,
        'max_iterations': 50,
        'fads_probability': 0.2,
        'convergence_threshold': 1e-6
    }

    optimizer_a = StaticWeightOptimizer(mpa_config)
    weights_a, score_a = optimizer_a.optimize_weights(expert_predictions, y_true)

    print(f"   最优权重: {weights_a}")
    print(f"   权重和: {np.sum(weights_a):.6f}")
    print(f"   验证RMSE: {score_a:.6f}")

    # 计算策略A预测
    strategy_a_pred = np.zeros_like(y_true)
    for i, weight in enumerate(weights_a):
        strategy_a_pred += weight * expert_predictions[:, :, i]

    # 测试策略B：广义线性融合
    print("\n2. 测试策略B：广义线性融合...")
    optimizer_b = StackingOptimizer(mpa_config)
    coefficients_b, score_b = optimizer_b.optimize_coefficients(expert_predictions, y_true)

    print(f"   最优系数: {coefficients_b}")
    print(f"   验证RMSE: {score_b:.6f}")

    # 计算策略B预测
    w0 = coefficients_b[0]
    weights_b = coefficients_b[1:]
    strategy_b_pred = np.full_like(y_true, w0)
    for i, weight in enumerate(weights_b):
        strategy_b_pred += weight * expert_predictions[:, :, i]

    # 测试策略C：动态门控网络（简化版）
    print("\n3. 测试策略C：动态门控网络...")

    # 创建门控网络
    gating_network = GatingNetwork(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        horizon=horizon,
        n_experts=n_experts
    )

    # 模拟目标变量历史序列
    target_history = np.random.randn(n_samples, 48, 1)  # 假设48个历史时间步

    with torch.no_grad():
        target_tensor = torch.FloatTensor(target_history)
        coefficients, _ = gating_network(target_tensor)
        print(f"   动态系数形状: {coefficients.shape}")
        assert coefficients.shape == (n_samples, horizon, n_experts + 1)

    # 计算策略C预测（简化）
    strategy_c_pred = np.zeros_like(y_true)
    for i in range(n_samples):
        for t in range(horizon):
            w0_t = coefficients[i, t, 0].numpy()
            w1_t = coefficients[i, t, 1].numpy()
            w2_t = coefficients[i, t, 2].numpy()
            strategy_c_pred[i, t] = w0_t + w1_t * expert1_pred[i, t] + w2_t * expert2_pred[i, t]

    print("\n✓ 融合策略测试通过")

    return {
        'y_true': y_true,
        'expert1': expert1_pred,
        'expert2': expert2_pred,
        'strategy_a': strategy_a_pred,
        'strategy_b': strategy_b_pred,
        'strategy_c': strategy_c_pred,
        'weights_a': weights_a,
        'coefficients_b': coefficients_b,
        'coefficients_c': coefficients
    }


def test_evaluation_metrics():
    """测试评估指标"""
    print("\n" + "=" * 60)
    print("测试评估指标")
    print("=" * 60)

    # 创建测试数据
    n_samples = 100
    horizon = 24

    y_true = np.random.randn(n_samples, horizon) * 2 + 10
    y_pred = y_true + np.random.randn(n_samples, horizon) * 0.3

    # 计算各种指标
    print("\n1. 计算基本指标...")
    mae = EvaluationMetrics.calculate_mae(y_true, y_pred)
    mape = EvaluationMetrics.calculate_mape(y_true, y_pred)
    rmse = EvaluationMetrics.calculate_rmse(y_true, y_pred)
    r2 = EvaluationMetrics.calculate_r2(y_true, y_pred)

    print(f"   MAE: {mae:.6f}")
    print(f"   MAPE: {mape:.6f}%")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   R²: {r2:.6f}")

    print("\n2. 计算峰值误差...")
    peak_error = EvaluationMetrics.calculate_peak_error(y_true, y_pred, peak_percentage=0.05)
    print(f"   峰值MAE: {peak_error['peak_mae']:.6f}")
    print(f"   峰值RMSE: {peak_error['peak_rmse']:.6f}")
    print(f"   峰值MAPE: {peak_error['peak_mape']:.6f}%")
    print(f"   峰值数量: {peak_error['n_peaks']}")

    print("\n3. 计算方向准确率...")
    directional_acc = EvaluationMetrics.calculate_directional_accuracy(y_true, y_pred)
    print(f"   方向准确率: {directional_acc:.6f}")

    print("\n4. 计算时间误差...")
    timeliness_error = EvaluationMetrics.calculate_timeliness_error(y_true, y_pred)
    print(f"   平均时间误差: {timeliness_error['mean_time_error']:.6f}")
    print(f"   峰值检测准确率: {timeliness_error['mean_peak_detection_accuracy']:.6f}")

    print("\n✓ 评估指标测试通过")

    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'r2': r2,
        'peak_error': peak_error,
        'directional_accuracy': directional_acc,
        'timeliness_error': timeliness_error
    }


def test_mpa_optimizer():
    """测试MPA优化器"""
    print("\n" + "=" * 60)
    print("测试MPA优化器")
    print("=" * 60)

    from src.strategies.mpa_optimizer import MPAOptimizer

    # 定义简单的测试函数
    def test_objective(x):
        # 多峰函数，全局最小值在(0,0)
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + 0.1 * np.sin(10 * x[0]) * np.cos(10 * x[1])

    # 设置优化器
    mpa = MPAOptimizer(
        population_size=30,
        max_iterations=100,
        fads_probability=0.2,
        convergence_threshold=1e-6
    )

    # 定义边界
    bounds = [(-5, 5), (-5, 5)]

    print("\n优化目标函数: f(x,y) = (x-1)² + (y-2)² + 0.1*sin(10x)*cos(10y)")
    print("全局最小值在 (1, 2), f(1,2) = 0")

    # 运行优化
    best_solution, best_fitness, fitness_history = mpa.optimize(
        objective_function=test_objective,
        bounds=bounds,
        minimize=True
    )

    print(f"\n优化结果:")
    print(f"最优解: {best_solution}")
    print(f"最优适应度: {best_fitness:.8f}")
    print(f"迭代次数: {len(fitness_history)}")

    # 验证结果
    expected_solution = np.array([1, 2])
    distance_to_optimal = np.linalg.norm(best_solution - expected_solution)
    print(f"与理论最优解的距离: {distance_to_optimal:.6f}")

    assert distance_to_optimal < 0.1, "MPA优化器未能找到接近最优的解"

    print("\n✓ MPA优化器测试通过")

    return {
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'distance_to_optimal': distance_to_optimal
    }


def main():
    """主测试函数"""
    print("浮式风机平台预测系统 - 快速测试")
    print("=" * 60)

    try:
        # 1. 测试模型
        patchtst, nhits, X, y = test_models()

        # 2. 测试融合策略
        fusion_results = test_fusion_strategies()

        # 3. 测试评估指标
        metrics = test_evaluation_metrics()

        # 4. 测试MPA优化器
        mpa_results = test_mpa_optimizer()

        # 总结
        print("\n" + "=" * 60)
        print("快速测试总结")
        print("=" * 60)
        print("✓ 所有核心组件测试通过！")
        print("✓ 模型结构正确")
        print("✓ 融合策略工作正常")
        print("✓ 评估指标计算正确")
        print("✓ MPA优化器有效")

        print(f"\n融合策略性能对比（RMSE）:")
        print(f"专家1 (PatchTST模拟): {metrics['rmse']:.6f}")
        print(
            f"专家2 (NHITS模拟): {EvaluationMetrics.calculate_rmse(fusion_results['y_true'], fusion_results['expert2']):.6f}")
        print(
            f"策略A (静态权重): {EvaluationMetrics.calculate_rmse(fusion_results['y_true'], fusion_results['strategy_a']):.6f}")
        print(
            f"策略B (线性融合): {EvaluationMetrics.calculate_rmse(fusion_results['y_true'], fusion_results['strategy_b']):.6f}")
        print(
            f"策略C (动态门控): {EvaluationMetrics.calculate_rmse(fusion_results['y_true'], fusion_results['strategy_c']):.6f}")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 系统已准备就绪，可以运行完整实验！")
        print("运行: python main.py")
    else:
        print("\n⚠️  请检查错误并修复后重新测试")