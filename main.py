"""
浮式风机平台运动响应预测主程序
集成三种融合策略的完整实验流程
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
import logging
import joblib
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_preprocessing.data_loader import DataLoader
from src.models.patchtst import PatchTST, PatchTSTTrainer
from src.models.nhits import NHITS, NHITSTrainer
from src.strategies.mpa_optimizer import StaticWeightOptimizer, StackingOptimizer
from src.strategies.gating_network import DynamicFusionModel, DynamicFusionTrainer, GatingNetwork
from src.evaluation.metrics import EvaluationMetrics, ModelComparison
from src.utils.hyperparameter_optimization import HyperparameterOptimizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('floating_wind_platform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FloatingWindPlatformExperiment:
    """浮式风机平台实验主类"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        初始化实验

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        # 创建输出目录
        self.results_dir = self.config['output']['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)

        # 初始化组件
        self.data_loader = None
        self.expert_models = {}
        self.fusion_models = {}
        self.results = {}

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        logger.info("开始数据加载和预处理...")

        # 初始化数据加载器
        self.data_loader = DataLoader(self.config_path)

        # 加载数据
        data = self.data_loader.load_data()

        # 预处理数据
        target_scaled, covariate_scaled = self.data_loader.preprocess_data()

        # 创建序列样本
        X, y = self.data_loader.create_sequences(target_scaled, covariate_scaled)

        # 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(X, y)

        # 保存处理后的数据
        self.data_loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)

        # 保存为类属性
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        logger.info(f"数据预处理完成:")
        logger.info(f"训练集: {X_train.shape}, {y_train.shape}")
        logger.info(f"验证集: {X_val.shape}, {y_val.shape}")
        logger.info(f"测试集: {X_test.shape}, {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def optimize_expert_models(self):
        """优化专家模型超参数"""
        logger.info("开始专家模型超参数优化...")

        # 初始化超参数优化器
        hp_optimizer = HyperparameterOptimizer(self.config_path)

        # 优化PatchTST
        logger.info("优化PatchTST超参数...")
        patchtst_params = hp_optimizer.optimize_patchtst(
            self.X_train, self.y_train, self.X_val, self.y_val
        )

        # 优化NHITS
        logger.info("优化NHITS超参数...")
        nhits_params = hp_optimizer.optimize_nhits(
            self.X_train, self.y_train, self.X_val, self.y_val
        )

        # 保存优化结果
        self.optimal_params = {
            'patchtst': patchtst_params,
            'nhits': nhits_params
        }

        # 保存到文件
        with open(f"{self.results_dir}/optimal_hyperparameters.yaml", 'w') as f:
            yaml.dump(self.optimal_params, f)

        logger.info("超参数优化完成")

        return self.optimal_params

    def train_expert_models(self, use_optimized_params: bool = True):
        """训练专家模型"""
        logger.info("开始训练专家模型...")

        n_samples, input_size, num_features = self.X_train.shape
        _, horizon = self.y_train.shape

        # 使用优化的超参数或默认配置
        if use_optimized_params and hasattr(self, 'optimal_params'):
            patchtst_config = self.optimal_params['patchtst']
            nhits_config = self.optimal_params['nhits']
        else:
            patchtst_config = self.config['models']['patchtst']
            nhits_config = self.config['models']['nhits']

        # 创建PatchTST模型
        logger.info("创建PatchTST模型...")
        patchtst_model = PatchTST(
            input_size=input_size,
            horizon=horizon,
            num_features=num_features,
            **patchtst_config
        )

        # 创建NHITS模型
        logger.info("创建NHITS模型...")
        nhits_model = NHITS(
            input_size=input_size,
            horizon=horizon,
            num_features=num_features,
            **nhits_config
        )

        # 训练PatchTST
        logger.info("训练PatchTST...")
        patchtst_trainer = PatchTSTTrainer(patchtst_model, self.device)
        learning_rate = patchtst_config.get('learning_rate', 1e-3)
        patchtst_trainer.setup_training(learning_rate=learning_rate)

        # 创建数据加载器
        train_loader = self._create_dataloader(self.X_train, self.y_train)
        val_loader = self._create_dataloader(self.X_val, self.y_val, shuffle=False)

        patchtst_trainer.train_model(train_loader, val_loader, num_epochs=self.config['training']['num_epochs'])

        # 训练NHITS
        logger.info("训练NHITS...")
        nhits_trainer = NHITSTrainer(nhits_model, self.device)
        learning_rate = nhits_config.get('learning_rate', 1e-3)
        nhits_trainer.setup_training(learning_rate=learning_rate)

        nhits_trainer.train_model(train_loader, val_loader, num_epochs=self.config['training']['num_epochs'])

        # 保存模型
        self.expert_models = {
            'patchtst': {'model': patchtst_model, 'trainer': patchtst_trainer},
            'nhits': {'model': nhits_model, 'trainer': nhits_trainer}
        }

        # 保存模型到文件
        patchtst_trainer.save_model(f"{self.results_dir}/patchtst_model.pth")
        nhits_trainer.save_model(f"{self.results_dir}/nhits_model.pth")

        logger.info("专家模型训练完成")

        return self.expert_models

    def get_expert_predictions(self):
        """获取专家模型在测试集上的预测"""
        logger.info("获取专家模型预测...")

        test_loader = self._create_dataloader(self.X_test, self.y_test, shuffle=False)

        expert_predictions = {}

        for name, expert_data in self.expert_models.items():
            trainer = expert_data['trainer']
            predictions = trainer.predict(test_loader)
            expert_predictions[name] = predictions

        # 保存预测结果
        self.expert_predictions = expert_predictions

        # 保存到文件
        for name, predictions in expert_predictions.items():
            np.save(f"{self.results_dir}/{name}_predictions.npy", predictions)

        logger.info("专家预测获取完成")

        return expert_predictions

    def implement_strategy_a(self):
        """实现策略A：静态优化权重"""
        logger.info("实现策略A：静态优化权重...")

        # 准备数据
        patchtst_pred = self.expert_predictions['patchtst']
        nhits_pred = self.expert_predictions['nhits']

        # 合并预测 [n_samples, horizon, n_experts]
        expert_preds_combined = np.stack([patchtst_pred, nhits_pred], axis=2)

        # 使用验证集来优化权重
        val_loader = self._create_dataloader(self.X_val, self.y_val, shuffle=False)

        val_predictions = {}
        for name, expert_data in self.expert_models.items():
            trainer = expert_data['trainer']
            predictions = trainer.predict(val_loader)
            val_predictions[name] = predictions

        val_expert_preds_combined = np.stack([val_predictions['patchtst'], val_predictions['nhits']], axis=2)

        # 使用MPA优化权重
        mpa_config = self.config['mpa']
        optimizer = StaticWeightOptimizer(mpa_config)

        optimal_weights, best_score = optimizer.optimize_weights(
            val_expert_preds_combined, self.y_val
        )

        # 在测试集上应用权重
        strategy_a_predictions = np.zeros_like(self.y_test)
        for i, weight in enumerate(optimal_weights):
            strategy_a_predictions += weight * expert_preds_combined[:, :, i]

        # 保存结果
        self.strategy_a_results = {
            'predictions': strategy_a_predictions,
            'weights': optimal_weights,
            'validation_score': best_score
        }

        np.save(f"{self.results_dir}/strategy_a_predictions.npy", strategy_a_predictions)
        np.save(f"{self.results_dir}/strategy_a_weights.npy", optimal_weights)

        logger.info(f"策略A完成 - 最优权重: {optimal_weights}, 验证集RMSE: {best_score:.6f}")

        return self.strategy_a_results

    def implement_strategy_b(self):
        """实现策略B：广义线性融合"""
        logger.info("实现策略B：广义线性融合...")

        # 准备数据
        patchtst_pred = self.expert_predictions['patchtst']
        nhits_pred = self.expert_predictions['nhits']
        expert_preds_combined = np.stack([patchtst_pred, nhits_pred], axis=2)

        # 使用验证集来优化系数
        val_loader = self._create_dataloader(self.X_val, self.y_val, shuffle=False)

        val_predictions = {}
        for name, expert_data in self.expert_models.items():
            trainer = expert_data['trainer']
            predictions = trainer.predict(val_loader)
            val_predictions[name] = predictions

        val_expert_preds_combined = np.stack([val_predictions['patchtst'], val_predictions['nhits']], axis=2)

        # 使用MPA优化线性系数
        mpa_config = self.config['mpa']
        optimizer = StackingOptimizer(mpa_config)

        optimal_coefficients, best_score = optimizer.optimize_coefficients(
            val_expert_preds_combined, self.y_val
        )

        # 在测试集上应用线性组合
        w0 = optimal_coefficients[0]  # 截距
        weights = optimal_coefficients[1:]  # 专家权重

        strategy_b_predictions = np.full_like(self.y_test, w0)
        for i, weight in enumerate(weights):
            strategy_b_predictions += weight * expert_preds_combined[:, :, i]

        # 保存结果
        self.strategy_b_results = {
            'predictions': strategy_b_predictions,
            'coefficients': optimal_coefficients,
            'validation_score': best_score
        }

        np.save(f"{self.results_dir}/strategy_b_predictions.npy", strategy_b_predictions)
        np.save(f"{self.results_dir}/strategy_b_coefficients.npy", optimal_coefficients)

        logger.info(f"策略B完成 - 最优系数: {optimal_coefficients}, 验证集RMSE: {best_score:.6f}")

        return self.strategy_b_results

    def implement_strategy_c(self):
        """实现策略C：动态门控网络"""
        logger.info("实现策略C：动态门控网络...")

        n_samples, input_size, num_features = self.X_train.shape
        _, horizon = self.y_train.shape

        # 创建门控网络
        gating_config = self.config['models']['gating_network']
        gating_network = GatingNetwork(
            input_size=1,  # 只处理目标变量
            horizon=horizon,
            n_experts=2,  # PatchTST和NHITS
            **gating_config
        )

        # 提取专家模型
        patchtst_model = self.expert_models['patchtst']['model']
        nhits_model = self.expert_models['nhits']['model']
        expert_models_list = [patchtst_model, nhits_model]

        # 创建动态融合模型
        fusion_model = DynamicFusionModel(
            gating_network=gating_network,
            input_size=input_size,
            horizon=horizon,
            n_experts=2,
            num_features=num_features
        )

        # 创建训练器
        trainer = DynamicFusionTrainer(fusion_model, expert_models_list, self.device)
        trainer.setup_training()

        # 创建数据加载器
        train_loader = self._create_dataloader(self.X_train, self.y_train)
        val_loader = self._create_dataloader(self.X_val, self.y_val, shuffle=False)
        test_loader = self._create_dataloader(self.X_test, self.y_test, shuffle=False)

        # 训练模型
        trainer.train_model(train_loader, val_loader, num_epochs=self.config['training']['num_epochs'])

        # 预测
        strategy_c_predictions, coefficients = trainer.predict(test_loader)

        # 保存结果
        self.strategy_c_results = {
            'predictions': strategy_c_predictions,
            'coefficients': coefficients,
            'model': fusion_model,
            'trainer': trainer
        }

        np.save(f"{self.results_dir}/strategy_c_predictions.npy", strategy_c_predictions)
        np.save(f"{self.results_dir}/strategy_c_coefficients.npy", coefficients)

        # 保存模型
        trainer.save_model(f"{self.results_dir}/strategy_c_model.pth")

        logger.info("策略C完成")

        return self.strategy_c_results

    def evaluate_all_strategies(self):
        """评估所有策略的性能"""
        logger.info("评估所有策略性能...")

        # 评估专家模型
        expert_metrics = {}
        for name, predictions in self.expert_predictions.items():
            metrics = EvaluationMetrics.calculate_all_metrics(
                self.y_test, predictions,
                peak_percentage=self.config['evaluation']['peak_percentage']
            )
            expert_metrics[name] = metrics

        # 评估策略A
        strategy_a_metrics = EvaluationMetrics.calculate_all_metrics(
            self.y_test, self.strategy_a_results['predictions'],
            peak_percentage=self.config['evaluation']['peak_percentage']
        )

        # 评估策略B
        strategy_b_metrics = EvaluationMetrics.calculate_all_metrics(
            self.y_test, self.strategy_b_results['predictions'],
            peak_percentage=self.config['evaluation']['peak_percentage']
        )

        # 评估策略C
        strategy_c_metrics = EvaluationMetrics.calculate_all_metrics(
            self.y_test, self.strategy_c_results['predictions'],
            peak_percentage=self.config['evaluation']['peak_percentage']
        )

        # 合并所有结果
        all_results = {
            **expert_metrics,
            'Strategy_A_Static': strategy_a_metrics,
            'Strategy_B_Stacking': strategy_b_metrics,
            'Strategy_C_Dynamic': strategy_c_metrics
        }

        self.all_metrics = all_results

        # 保存结果
        with open(f"{self.results_dir}/all_metrics.yaml", 'w') as f:
            # 转换numpy类型为Python原生类型
            yaml_safe_results = {}
            for model_name, metrics in all_results.items():
                yaml_safe_results[model_name] = {}
                for metric_name, value in metrics.items():
                    if isinstance(value, (np.ndarray, np.generic)):
                        yaml_safe_results[model_name][metric_name] = float(value)
                    elif isinstance(value, (list, tuple)):
                        yaml_safe_results[model_name][metric_name] = [
                            float(v) if isinstance(v, (np.ndarray, np.generic)) else v for v in value]
                    else:
                        yaml_safe_results[model_name][metric_name] = value

            yaml.dump(yaml_safe_results, f)

        # 比较模型
        comparison = ModelComparison.compare_models(all_results)

        with open(f"{self.results_dir}/model_comparison.yaml", 'w') as f:
            yaml.dump(comparison, f)

        logger.info("性能评估完成")

        return all_results, comparison

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True, batch_size: Optional[int] = None):
        """创建数据加载器"""
        if batch_size is None:
            batch_size = self.config['training']['batch_size']

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def generate_summary_report(self):
        """生成总结报告"""
        logger.info("生成总结报告...")

        # 创建结果摘要
        summary = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'config_file': self.config_path,
                'device': str(self.device),
                'data_shape': {
                    'train': f"{self.X_train.shape}, {self.y_train.shape}",
                    'val': f"{self.X_val.shape}, {self.y_val.shape}",
                    'test': f"{self.X_test.shape}, {self.y_test.shape}"
                }
            },
            'performance_summary': {}
        }

        # 性能摘要
        for model_name, metrics in self.all_metrics.items():
            summary['performance_summary'][model_name] = {
                'RMSE': float(metrics['RMSE']),
                'MAE': float(metrics['MAE']),
                'MAPE': float(metrics['MAPE']),
                'R2': float(metrics['R2']),
                'Peak_RMSE': float(metrics['peak_rmse'])
            }

        # 保存摘要
        with open(f"{self.results_dir}/experiment_summary.yaml", 'w') as f:
            yaml.dump(summary, f)

        logger.info("总结报告生成完成")

        return summary

    def run_complete_experiment(self, optimize_hyperparameters: bool = True):
        """运行完整实验"""
        logger.info("开始完整实验流程...")

        try:
            # 1. 数据预处理
            self.load_and_preprocess_data()

            # 2. 超参数优化（可选）
            if optimize_hyperparameters:
                self.optimize_expert_models()

            # 3. 训练专家模型
            self.train_expert_models(use_optimized_params=optimize_hyperparameters)

            # 4. 获取专家预测
            self.get_expert_predictions()

            # 5. 实现三种融合策略
            self.implement_strategy_a()
            self.implement_strategy_b()
            self.implement_strategy_c()

            # 6. 评估所有策略
            self.evaluate_all_strategies()

            # 7. 生成总结报告
            self.generate_summary_report()

            logger.info("完整实验流程完成！")

            return self.all_metrics

        except Exception as e:
            logger.error(f"实验过程中出现错误: {str(e)}")
            raise


def main():
    """主函数"""
    # 创建实验实例
    experiment = FloatingWindPlatformExperiment()

    # 运行完整实验
    results = experiment.run_complete_experiment(optimize_hyperparameters=True)

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("实验结果摘要")
    print("=" * 60)

    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  MAPE: {metrics['MAPE']:.6f}%")
        print(f"  R²:   {metrics['R2']:.6f}")
        print(f"  峰值RMSE: {metrics['peak_rmse']:.6f}")

    print(f"\n详细结果已保存到: {experiment.results_dir}")


if __name__ == "__main__":
    main()