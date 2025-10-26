"""
评估指标计算模块
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """评估指标计算器"""

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算平均绝对误差 (MAE)"""
        return mean_absolute_error(y_true.flatten(), y_pred.flatten())

    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算平均绝对百分比误差 (MAPE)"""
        # 避免除零错误
        mask = np.abs(y_true) > 1e-10
        if np.sum(mask) == 0:
            return np.inf

        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算均方根误差 (RMSE)"""
        return np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算决定系数 (R²)"""
        return r2_score(y_true.flatten(), y_pred.flatten())

    @staticmethod
    def calculate_peak_error(y_true: np.ndarray, y_pred: np.ndarray, peak_percentage: float = 0.05) -> Dict:
        """
        计算前百分之几最大波峰的预测误差

        Args:
            y_true: 真实值 [n_samples, horizon]
            y_pred: 预测值 [n_samples, horizon]
            peak_percentage: 峰值百分比 (0.05 = 前5%)

        Returns:
            峰值误差统计
        """
        n_samples, horizon = y_true.shape

        # 合并所有时间步
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # 找到前百分之几的最大波峰
        n_peaks = int(len(y_true_flat) * peak_percentage)

        # 找到真实值中的峰值索引
        peak_indices = np.argpartition(y_true_flat, -n_peaks)[-n_peaks:]

        # 计算峰值误差
        peak_true = y_true_flat[peak_indices]
        peak_pred = y_pred_flat[peak_indices]

        peak_mae = np.mean(np.abs(peak_true - peak_pred))
        peak_rmse = np.sqrt(np.mean((peak_true - peak_pred) ** 2))
        peak_mape = np.mean(np.abs((peak_true - peak_pred) / peak_true)) * 100

        return {
            'peak_mae': peak_mae,
            'peak_rmse': peak_rmse,
            'peak_mape': peak_mape,
            'n_peaks': n_peaks,
            'peak_indices': peak_indices,
            'peak_true_values': peak_true,
            'peak_pred_values': peak_pred
        }

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, peak_percentage: float = 0.05) -> Dict:
        """
        计算所有评估指标

        Args:
            y_true: 真实值 [n_samples, horizon]
            y_pred: 预测值 [n_samples, horizon]
            peak_percentage: 峰值百分比

        Returns:
            所有评估指标
        """
        metrics = {
            'MAE': EvaluationMetrics.calculate_mae(y_true, y_pred),
            'MAPE': EvaluationMetrics.calculate_mape(y_true, y_pred),
            'RMSE': EvaluationMetrics.calculate_rmse(y_true, y_pred),
            'R2': EvaluationMetrics.calculate_r2(y_true, y_pred)
        }

        # 计算峰值误差
        peak_metrics = EvaluationMetrics.calculate_peak_error(y_true, y_pred, peak_percentage)
        metrics.update(peak_metrics)

        return metrics

    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算方向准确率（预测变化方向的准确性）

        Args:
            y_true: 真实值 [n_samples, horizon]
            y_pred: 预测值 [n_samples, horizon]

        Returns:
            方向准确率 (0-1)
        """
        # 计算相邻时间步的变化
        true_changes = np.diff(y_true, axis=1)
        pred_changes = np.diff(y_pred, axis=1)

        # 计算方向一致性
        same_direction = np.sign(true_changes) == np.sign(pred_changes)
        directional_accuracy = np.mean(same_direction)

        return directional_accuracy

    @staticmethod
    def calculate_timeliness_error(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.1) -> Dict:
        """
        计算时间误差（峰值预测的时间偏移）

        Args:
            y_true: 真实值 [n_samples, horizon]
            y_pred: 预测值 [n_samples, horizon]
            threshold: 峰值阈值

        Returns:
            时间误差统计
        """
        n_samples, horizon = y_true.shape

        time_errors = []
        peak_detection_accuracy = []

        for i in range(n_samples):
            true_series = y_true[i]
            pred_series = y_pred[i]

            # 检测峰值（超过均值+阈值*标准差）
            true_threshold = np.mean(true_series) + threshold * np.std(true_series)
            pred_threshold = np.mean(pred_series) + threshold * np.std(pred_series)

            true_peaks = np.where(true_series > true_threshold)[0]
            pred_peaks = np.where(pred_series > pred_threshold)[0]

            if len(true_peaks) > 0 and len(pred_peaks) > 0:
                # 计算时间误差
                for true_peak in true_peaks:
                    if len(pred_peaks) > 0:
                        # 找到最近的预测峰值
                        closest_pred_idx = np.argmin(np.abs(pred_peaks - true_peak))
                        time_error = pred_peaks[closest_pred_idx] - true_peak
                        time_errors.append(time_error)

                # 峰值检测准确率
                true_positives = len(set(true_peaks) & set(pred_peaks))
                false_positives = len(set(pred_peaks) - set(true_peaks))
                false_negatives = len(set(true_peaks) - set(pred_peaks))

                if (true_positives + false_positives + false_negatives) > 0:
                    accuracy = true_positives / (true_positives + false_positives + false_negatives)
                    peak_detection_accuracy.append(accuracy)

        return {
            'mean_time_error': np.mean(time_errors) if time_errors else 0,
            'std_time_error': np.std(time_errors) if time_errors else 0,
            'mean_peak_detection_accuracy': np.mean(peak_detection_accuracy) if peak_detection_accuracy else 0,
            'time_errors': time_errors,
            'peak_detection_accuracy': peak_detection_accuracy
        }


class ModelComparison:
    """模型比较工具"""

    @staticmethod
    def compare_models(results: Dict[str, Dict]) -> Dict:
        """
        比较多个模型的性能

        Args:
            results: {model_name: metrics_dict}

        Returns:
            比较结果
        """
        comparison = {
            'rankings': {},
            'best_model': {},
            'relative_improvement': {}
        }

        models = list(results.keys())
        metrics = ['MAE', 'MAPE', 'RMSE']

        # 为每个指标排序模型
        for metric in metrics:
            if metric in results[models[0]]:
                model_scores = [(model, results[model][metric]) for model in models]

                # 对于MAE、MAPE、RMSE，越小越好
                model_scores.sort(key=lambda x: x[1])

                comparison['rankings'][metric] = {
                    'ranking': [(i + 1, model, score) for i, (model, score) in enumerate(model_scores)],
                    'best_model': model_scores[0][0],
                    'best_score': model_scores[0][1]
                }

        # 找到综合最佳模型
        if 'RMSE' in results[models[0]]:
            best_overall = comparison['rankings']['RMSE']['best_model']
            comparison['best_model']['overall'] = {
                'model': best_overall,
                'rmse': results[best_overall]['RMSE']
            }

        # 计算相对改进
        if len(models) >= 2:
            baseline_model = models[0]  # 第一个模型作为基线
            baseline_rmse = results[baseline_model]['RMSE']

            comparison['relative_improvement'][baseline_model] = {}
            for model in models[1:]:
                model_rmse = results[model]['RMSE']
                improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
                comparison['relative_improvement'][model] = {
                    'baseline': baseline_model,
                    'improvement_percentage': improvement
                }

        return comparison


class CrossValidationEvaluator:
    """交叉验证评估器"""

    @staticmethod
    def time_series_split(data_size: int, n_splits: int = 5, test_size: float = 0.2):
        """
        时间序列交叉验证分割

        Args:
            data_size: 数据总大小
            n_splits: 分割数
            test_size: 测试集比例

        Returns:
            训练集和测试集索引
        """
        test_length = int(data_size * test_size)
        train_length = data_size - test_length

        splits = []

        for i in range(n_splits):
            # 逐步增加训练集大小
            train_end = int(train_length * (i + 1) / n_splits)
            test_end = min(train_end + test_length, data_size)

            train_idx = np.arange(train_end)
            test_idx = np.arange(train_end, test_end)

            if len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    @staticmethod
    def cross_validate(model_class, model_config, X, y, n_splits=5, **fit_params):
        """
        执行交叉验证

        Args:
            model_class: 模型类
            model_config: 模型配置
            X: 特征数据
            y: 目标数据
            n_splits: 分割数
            **fit_params: 拟合参数

        Returns:
            交叉验证结果
        """
        splits = CrossValidationEvaluator.time_series_split(len(X), n_splits)

        cv_results = {
            'splits': [],
            'mean_metrics': {},
            'std_metrics': {}
        }

        all_metrics = []

        for i, (train_idx, test_idx) in enumerate(splits):
            # 分割数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 创建和训练模型
            model = model_class(**model_config)

            # 这里需要实现模型训练逻辑
            # model.fit(X_train, y_train, **fit_params)
            # y_pred = model.predict(X_test)

            # 计算指标
            # metrics = EvaluationMetrics.calculate_all_metrics(y_test, y_pred)
            # all_metrics.append(metrics)

            # cv_results['splits'].append({
            #     'split': i,
            #     'metrics': metrics,
            #     'n_train': len(X_train),
            #     'n_test': len(X_test)
            # })

        # 计算平均指标
        if all_metrics:
            metric_names = all_metrics[0].keys()

            for metric_name in metric_names:
                values = [m[metric_name] for m in all_metrics]
                cv_results['mean_metrics'][metric_name] = np.mean(values)
                cv_results['std_metrics'][metric_name] = np.std(values)

        return cv_results


def main():
    """测试评估指标"""
    np.random.seed(42)

    # 创建测试数据
    n_samples = 1000
    horizon = 24

    y_true = np.random.randn(n_samples, horizon) * 2 + 10
    y_pred = y_true + np.random.randn(n_samples, horizon) * 0.5  # 添加一些噪声

    # 计算所有指标
    metrics = EvaluationMetrics.calculate_all_metrics(y_true, y_pred, peak_percentage=0.05)

    print("评估指标测试结果:")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"MAPE: {metrics['MAPE']:.6f}%")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R²: {metrics['R2']:.6f}")
    print(f"峰值MAE (前5%): {metrics['peak_mae']:.6f}")
    print(f"峰值RMSE (前5%): {metrics['peak_rmse']:.6f}")
    print(f"峰值MAPE (前5%): {metrics['peak_mape']:.6f}%")

    # 测试方向准确率
    directional_acc = EvaluationMetrics.calculate_directional_accuracy(y_true, y_pred)
    print(f"方向准确率: {directional_acc:.6f}")

    # 测试时间误差
    timeliness_error = EvaluationMetrics.calculate_timeliness_error(y_true, y_pred)
    print(f"平均时间误差: {timeliness_error['mean_time_error']:.6f}")
    print(f"峰值检测准确率: {timeliness_error['mean_peak_detection_accuracy']:.6f}")

    print("\n评估指标测试完成！")


if __name__ == "__main__":
    main()