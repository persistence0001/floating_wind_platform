"""
海洋捕食者算法 (MPA) 优化器
用于优化融合权重
"""

import numpy as np
from typing import Callable, Tuple, List
import logging

logger = logging.getLogger(__name__)


class MPAOptimizer:
    """海洋捕食者算法优化器"""

    def __init__(self,
                 population_size: int = 30,
                 max_iterations: int = 100,
                 fads_probability: float = 0.2,
                 convergence_threshold: float = 1e-6):
        """
        初始化MPA优化器

        Args:
            population_size: 种群大小
            max_iterations: 最大迭代次数
            fads_probability: FADs效应概率
            convergence_threshold: 收敛阈值
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.fads_probability = fads_probability
        self.convergence_threshold = convergence_threshold

    def optimize(self,
                 objective_function: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 constraint_function: Optional[Callable[[np.ndarray], bool]] = None,
                 minimize: bool = True) -> Tuple[np.ndarray, float, List[float]]:
        """
        优化目标函数

        Args:
            objective_function: 目标函数
            bounds: 变量边界 [(min, max), ...]
            constraint_function: 约束函数
            minimize: 是否最小化

        Returns:
            best_solution: 最优解
            best_fitness: 最优适应度
            fitness_history: 适应度历史
        """
        dim = len(bounds)

        # 初始化种群
        population = self._initialize_population(bounds)

        # 计算初始适应度
        fitness = np.array([self._evaluate_fitness(obj_func, ind, constraint_function)
                            for ind in population])

        # 找到初始最优解
        if minimize:
            best_idx = np.argmin(fitness)
        else:
            best_idx = np.argmax(fitness)

        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        fitness_history = [best_fitness]

        # 主循环
        for iteration in range(self.max_iterations):
            # 计算当前迭代的比例
            iteration_ratio = iteration / self.max_iterations

            # 三个阶段
            if iteration_ratio < 0.3:
                # 第一阶段：高速率探索
                population = self._phase1_exploration(population, bounds, iteration_ratio)
            elif iteration_ratio < 0.7:
                # 第二阶段：探索到开发的转换
                population = self._phase2_transition(population, best_solution, bounds, iteration_ratio)
            else:
                # 第三阶段：高速率开发
                population = self._phase3_exploitation(population, best_solution, bounds, iteration_ratio)

            # FADs效应
            population = self._apply_fads(population, bounds)

            # 边界检查
            population = self._boundary_check(population, bounds)

            # 重新计算适应度
            fitness = np.array([self._evaluate_fitness(obj_func, ind, constraint_function)
                                for ind in population])

            # 更新最优解
            if minimize:
                current_best_idx = np.argmin(fitness)
                if fitness[current_best_idx] < best_fitness:
                    best_solution = population[current_best_idx].copy()
                    best_fitness = fitness[current_best_idx]
            else:
                current_best_idx = np.argmax(fitness)
                if fitness[current_best_idx] > best_fitness:
                    best_solution = population[current_best_idx].copy()
                    best_fitness = fitness[current_best_idx]

            fitness_history.append(best_fitness)

            # 检查收敛
            if iteration > 10:
                recent_improvement = abs(fitness_history[-1] - fitness_history[-11])
                if recent_improvement < self.convergence_threshold:
                    logger.info(f"在第 {iteration} 代收敛")
                    break

            if iteration % 10 == 0:
                logger.info(f"迭代 {iteration}: 最优适应度 = {best_fitness:.6f}")

        return best_solution, best_fitness, fitness_history

    def _initialize_population(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """初始化种群"""
        dim = len(bounds)
        population = np.zeros((self.population_size, dim))

        for i in range(dim):
            low, high = bounds[i]
            population[:, i] = np.random.uniform(low, high, self.population_size)

        return population

    def _evaluate_fitness(self,
                          objective_function: Callable[[np.ndarray], float],
                          individual: np.ndarray,
                          constraint_function: Optional[Callable[[np.ndarray], bool]] = None) -> float:
        """评估适应度"""
        # 检查约束
        if constraint_function is not None and not constraint_function(individual):
            return np.inf  # 违反约束的解给予惩罚

        try:
            return objective_function(individual)
        except:
            return np.inf

    def _phase1_exploration(self, population: np.ndarray, bounds: List[Tuple[float, float]],
                            iteration_ratio: float) -> np.ndarray:
        """第一阶段：高速率探索"""
        new_population = population.copy()

        for i in range(self.population_size):
            # 随机选择捕食者
            predator_idx = np.random.randint(0, self.population_size)
            predator = population[predator_idx]

            # 计算步长
            step_size = 0.5 * np.random.randn(len(bounds)) * (1 - iteration_ratio)

            # 更新位置
            new_population[i] = population[i] + step_size * (predator - population[i])

        return new_population

    def _phase2_transition(self, population: np.ndarray, best_solution: np.ndarray,
                           bounds: List[Tuple[float, float]], iteration_ratio: float) -> np.ndarray:
        """第二阶段：探索到开发的转换"""
        new_population = population.copy()

        for i in range(self.population_size):
            if i < self.population_size // 2:
                # 精英个体向最优解移动
                step_size = 0.5 * np.random.randn(len(bounds)) * (1 - iteration_ratio)
                new_population[i] = best_solution + step_size * (best_solution - population[i])
            else:
                # 其他个体探索
                random_idx = np.random.randint(0, self.population_size)
                step_size = 0.5 * np.random.randn(len(bounds))
                new_population[i] = population[i] + step_size * (population[random_idx] - population[i])

        return new_population

    def _phase3_exploitation(self, population: np.ndarray, best_solution: np.ndarray,
                             bounds: List[Tuple[float, float]], iteration_ratio: float) -> np.ndarray:
        """第三阶段：高速率开发"""
        new_population = population.copy()

        for i in range(self.population_size):
            # 所有个体都向最优解移动
            step_size = 0.5 * np.random.randn(len(bounds)) * (1 - iteration_ratio)
            new_population[i] = best_solution + step_size * (best_solution - population[i])

        return new_population

    def _apply_fads(self, population: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """应用FADs效应"""
        new_population = population.copy()

        for i in range(self.population_size):
            if np.random.random() < self.fads_probability:
                # FADs效应：随机跳跃
                for j in range(len(bounds)):
                    low, high = bounds[j]
                    if np.random.random() < 0.5:
                        new_population[i, j] = np.random.uniform(low, high)

        return new_population

    def _boundary_check(self, population: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """边界检查"""
        for i in range(len(bounds)):
            low, high = bounds[i]
            population[:, i] = np.clip(population[:, i], low, high)

        return population


class StaticWeightOptimizer:
    """静态权重优化器"""

    def __init__(self, mpa_config: dict):
        """
        初始化静态权重优化器

        Args:
            mpa_config: MPA配置参数
        """
        self.mpa = MPAOptimizer(**mpa_config)

    def optimize_weights(self,
                         expert_predictions: np.ndarray,
                         true_values: np.ndarray,
                         minimize: bool = True) -> Tuple[np.ndarray, float]:
        """
        优化专家模型权重

        Args:
            expert_predictions: [n_samples, horizon, n_experts] 专家预测
            true_values: [n_samples, horizon] 真实值
            minimize: 是否最小化

        Returns:
            optimal_weights: 最优权重
            best_score: 最优分数
        """
        n_experts = expert_predictions.shape[2]

        def objective_function(weights: np.ndarray) -> float:
            """目标函数：RMSE"""
            # 确保权重和为1
            weights = weights / np.sum(weights)

            # 计算加权预测
            weighted_pred = np.zeros_like(true_values)
            for i in range(n_experts):
                weighted_pred += weights[i] * expert_predictions[:, :, i]

            # 计算RMSE
            rmse = np.sqrt(np.mean((weighted_pred - true_values) ** 2))
            return rmse

        def constraint_function(weights: np.ndarray) -> bool:
            """约束函数：权重非负且和为1"""
            return np.all(weights >= 0) and abs(np.sum(weights) - 1.0) < 1e-6

        # 定义边界 [0, 1]
        bounds = [(0.0, 1.0) for _ in range(n_experts)]

        # 优化
        optimal_weights, best_score, _ = self.mpa.optimize(
            objective_function=objective_function,
            bounds=bounds,
            constraint_function=constraint_function,
            minimize=minimize
        )

        # 归一化权重
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        return optimal_weights, best_score


class StackingOptimizer:
    """Stacking优化器（广义线性融合）"""

    def __init__(self, mpa_config: dict):
        """
        初始化Stacking优化器

        Args:
            mpa_config: MPA配置参数
        """
        self.mpa = MPAOptimizer(**mpa_config)

    def optimize_coefficients(self,
                              expert_predictions: np.ndarray,
                              true_values: np.ndarray,
                              minimize: bool = True) -> Tuple[np.ndarray, float]:
        """
        优化线性组合系数

        Args:
            expert_predictions: [n_samples, horizon, n_experts] 专家预测
            true_values: [n_samples, horizon] 真实值
            minimize: 是否最小化

        Returns:
            optimal_coefficients: 最优系数 [w0, w1, w2, ...]
            best_score: 最优分数
        """
        n_experts = expert_predictions.shape[2]
        n_coefficients = n_experts + 1  # 包括截距项

        def objective_function(coefficients: np.ndarray) -> float:
            """目标函数：RMSE"""
            w0 = coefficients[0]  # 截距
            weights = coefficients[1:]  # 专家权重

            # 计算线性组合预测
            weighted_pred = np.full_like(true_values, w0)
            for i in range(n_experts):
                weighted_pred += weights[i] * expert_predictions[:, :, i]

            # 计算RMSE
            rmse = np.sqrt(np.mean((weighted_pred - true_values) ** 2))
            return rmse

        # 定义边界 [-10, 10] 给更多自由度
        bounds = [(-10.0, 10.0) for _ in range(n_coefficients)]

        # 优化
        optimal_coefficients, best_score, _ = self.mpa.optimize(
            objective_function=objective_function,
            bounds=bounds,
            constraint_function=None,  # 无约束
            minimize=minimize
        )

        return optimal_coefficients, best_score


def main():
    """测试MPA优化器"""
    np.random.seed(42)

    # 创建测试数据
    n_samples = 1000
    horizon = 24
    n_experts = 2

    # 模拟专家预测
    expert1_pred = np.random.randn(n_samples, horizon) * 0.5 + 2.0
    expert2_pred = np.random.randn(n_samples, horizon) * 0.3 + 1.8
    expert_predictions = np.stack([expert1_pred, expert2_pred], axis=2)

    # 模拟真实值（更接近expert2）
    true_values = 1.5 + 0.8 * expert2_pred + np.random.randn(n_samples, horizon) * 0.2

    # 测试静态权重优化
    print("测试静态权重优化...")
    mpa_config = {
        'population_size': 20,
        'max_iterations': 50,
        'fads_probability': 0.2,
        'convergence_threshold': 1e-6
    }

    static_optimizer = StaticWeightOptimizer(mpa_config)
    optimal_weights, best_score = static_optimizer.optimize_weights(
        expert_predictions, true_values
    )

    print(f"最优权重: {optimal_weights}")
    print(f"最优RMSE: {best_score:.6f}")
    print(f"权重和: {np.sum(optimal_weights):.6f}")

    # 测试Stacking优化
    print("\n测试Stacking优化...")
    stacking_optimizer = StackingOptimizer(mpa_config)
    optimal_coefficients, best_score = stacking_optimizer.optimize_coefficients(
        expert_predictions, true_values
    )

    print(f"最优系数: {optimal_coefficients}")
    print(f"最优RMSE: {best_score:.6f}")

    print("\nMPA优化器测试完成！")


if __name__ == "__main__":
    main()
    