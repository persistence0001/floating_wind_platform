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
from typing import Optional

from src.strategies.mpa_optimizer import MPAOptimizer, StackingOptimizer
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_preprocessing.data_loader import DataLoader
from models.patchtst import PatchTST, PatchTSTTrainer
from models.nhits import NHITS, NHITSTrainer
from src.strategies.mpa_optimizer import MPAOptimizer,StackingOptimizer
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
        # 加载配置
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        # 创建输出目录
        self.results_dir = self.config['output']['results_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(self.results_dir, f"experiment_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

        # 创建可视化目录
        self.visualization_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(self.visualization_dir, exist_ok=True)

        # 初始化组件
        self.data_loader = None
        self.expert_models = {}
        self.fusion_models = {}
        self.results = {}
        self.time_stamps = None  # 用于存储时间戳
        # 新增：初始化所有关键属性为None，防止未初始化访问
        self.expert_predictions = None
        self.expert_predictions_original = None
        self.strategy_a_results = None
        self.strategy_b_results = None
        self.strategy_c_results = None
        self.y_test_original = None

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        logger.info("开始数据加载和预处理...")

        # 初始化数据加载器
        self.data_loader = DataLoader(self.config_path)

        # 加载数据
        data = self.data_loader.load_data()

        # 预处理数据

        

        target_scaled, covariate_scaled, time_stamps = self.data_loader.preprocess_data()
        
        self.time_stamps = time_stamps  # 保存时间戳

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
            self.X_train, self.y_train, self.X_val, self.y_val,
            params=self.config)                 # 把已有配置字典传进来

        # 优化NHITS
        logger.info("优化NHITS超参数...")
        nhits_params = hp_optimizer.optimize_nhits(
            self.X_train, self.y_train, self.X_val, self.y_val,
            params=self.config)                 # 把已有配置字典传进来        )

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

    def _create_dataloader(self, X, y, shuffle=True):
        """创建数据加载器"""
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return TorchDataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=shuffle
        )

    def train_expert_models(self, use_optimized_params: bool = True,
                            quick_mode: bool = False,
                            quick_epochs: Optional[int] = None):

        """训练专家模型

        Args:
            use_optimized_params: 是否使用优化的超参数
            quick_mode: 是否启用快速模式
            quick_epochs: 快速模式下的epoch数（如果为None则使用配置值）
        """
        logger.info("开始训练专家模型...")


        # 确定epoch数
        if quick_mode and quick_epochs is not None:
            num_epochs = quick_epochs
        else:
            num_epochs = self.config['training']['num_epochs']



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
            num_features=num_features, **patchtst_config
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

        patchtst_trainer.train_model(
            train_loader,
            val_loader,
            num_epochs=num_epochs,  # 使用变量而不是配置
            patience=self.config['training']['patience']
        )

        # 训练NHITS
        logger.info("训练NHITS...")
        nhits_trainer = NHITSTrainer(nhits_model, self.device)
        learning_rate = nhits_config.get('learning_rate', 1e-3)
        nhits_trainer.setup_training(learning_rate=learning_rate)

        nhits_trainer.train_model(
            train_loader,
            val_loader,
            num_epochs=num_epochs,  # 使用变量而不是配置
            patience=self.config['training']['patience']
        )

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

        # 保存预测结果（原始尺度和反标准化后的尺度）
        self.expert_predictions = expert_predictions

        # 反标准化预测结果
        self.expert_predictions_original = {}
        for name, preds in expert_predictions.items():
            # 保存标准化的预测
            np.save(f"{self.results_dir}/{name}_predictions_scaled.npy", preds)

            # 反标准化并保存
            original_preds = self._inverse_transform_predictions(preds)
            self.expert_predictions_original[name] = original_preds
            np.save(f"{self.results_dir}/{name}_predictions_original.npy", original_preds)

        logger.info("专家预测获取完成")

        return expert_predictions

    def _inverse_transform_predictions(self, scaled_preds):
        """将标准化的预测转换回原始尺度"""
        # scaled_preds 形状: [n_samples, horizon]
        n_samples, horizon = scaled_preds.shape
        original_preds = np.zeros_like(scaled_preds)

        for i in range(n_samples):
            for t in range(horizon):
                original_preds[i, t] = self.data_loader.inverse_transform_target(
                    scaled_preds[i, t].reshape(-1, 1)
                ).flatten()[0]

        return original_preds

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
        optimizer = MPAOptimizer.StaticWeightOptimizer(mpa_config)

        optimal_weights, best_score = optimizer.optimize_weights(
            val_expert_preds_combined, self.y_val
        )

        # 在测试集上应用权重
        strategy_a_predictions = np.zeros_like(self.y_test)
        for i, weight in enumerate(optimal_weights):
            strategy_a_predictions += weight * expert_preds_combined[:, :, i]

        # 反标准化预测结果
        strategy_a_predictions_original = self._inverse_transform_predictions(strategy_a_predictions)

        # 保存结果
        self.strategy_a_results = {
            'predictions': strategy_a_predictions,
            'predictions_original': strategy_a_predictions_original,
            'weights': optimal_weights,
            'validation_score': best_score
        }

        np.save(f"{self.results_dir}/strategy_a_predictions_scaled.npy", strategy_a_predictions)
        np.save(f"{self.results_dir}/strategy_a_predictions_original.npy", strategy_a_predictions_original)
        np.save(f"{self.results_dir}/strategy_a_weights.npy", optimal_weights)

        logger.info(f"策略A完成 - 最优权重: {optimal_weights}, 验证集RMSE: {best_score:.6f}")

        return self.strategy_a_results

    def implement_strategy_b(self):
        """实现策略B：广义线性融合"""
        logger.info("实现策略B：广义线性融合...")

        # 准备数据
        patchtst_pred = self.expert_predictions['patchtst']
        nhits_pred = self.expert_predictions['nhits']

        # 合并预测 [n_samples, horizon, n_experts]
        expert_preds_combined = np.stack([patchtst_pred, nhits_pred], axis=2)

        # 使用验证集来优化系数
        val_loader = self._create_dataloader(self.X_val, self.y_val, shuffle=False)

        val_predictions = {}
        for name, expert_data in self.expert_models.items():
            trainer = expert_data['trainer']
            predictions = trainer.predict(val_loader)
            val_predictions[name] = predictions

        val_expert_preds_combined = np.stack([val_predictions['patchtst'], val_predictions['nhits']], axis=2)

        # 使用MPA优化系数
        mpa_config = self.config['mpa']
        optimizer = StackingOptimizer(mpa_config)

        optimal_coefficients, best_score = optimizer.optimize_coefficients(
            val_expert_preds_combined, self.y_val
        )

        # 在测试集上应用系数
        # 系数包括截距项和权重
        w0 = optimal_coefficients[0]
        weights = optimal_coefficients[1:]

        strategy_b_predictions = np.full_like(self.y_test, w0)
        for i, weight in enumerate(weights):
            strategy_b_predictions += weight * expert_preds_combined[:, :, i]

        # 反标准化预测结果
        strategy_b_predictions_original = self._inverse_transform_predictions(strategy_b_predictions)

        # 保存结果
        self.strategy_b_results = {
            'predictions': strategy_b_predictions,
            'predictions_original': strategy_b_predictions_original,
            'coefficients': optimal_coefficients,
            'validation_score': best_score
        }

        np.save(f"{self.results_dir}/strategy_b_predictions_scaled.npy", strategy_b_predictions)
        np.save(f"{self.results_dir}/strategy_b_predictions_original.npy", strategy_b_predictions_original)
        np.save(f"{self.results_dir}/strategy_b_coefficients.npy", optimal_coefficients)

        logger.info(f"策略B完成 - 最优系数: {optimal_coefficients}, 验证集RMSE: {best_score:.6f}")

        return self.strategy_b_results

    def implement_strategy_c(self):
        """实现策略C：动态门控网络"""
        logger.info("实现策略C：动态门控网络...")

        # 获取专家模型
        patchtst_model = self.expert_models['patchtst']['model']
        nhits_model = self.expert_models['nhits']['model']

        # 创建动态融合模型
        n_experts = 2
        input_size = self.X_train.shape[1]
        horizon = self.y_train.shape[1]
        num_features = self.X_train.shape[2]

        # 创建门控网络
        gating_network = GatingNetwork(
            input_size=num_features,
            hidden_size=self.config['models']['gating_network']['hidden_size'],
            num_layers=self.config['models']['gating_network']['num_layers'],
            horizon=horizon,
            n_experts=n_experts,
            dropout=self.config['models']['gating_network']['dropout']
        )

        # 创建动态融合模型
        dynamic_model = DynamicFusionModel(
            gating_network=gating_network,
            input_size=input_size,
            horizon=horizon,
            n_experts=n_experts
        )

        # 创建训练器
        trainer = DynamicFusionTrainer(
            model=dynamic_model,
            expert_models=[patchtst_model, nhits_model],
            device=self.device
        )

        # 设置训练参数
        trainer.setup_training(
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # 创建数据加载器
        train_loader = self._create_dataloader(self.X_train, self.y_train)
        val_loader = self._create_dataloader(self.X_val, self.y_val, shuffle=False)

        # 训练模型
        logger.info("训练动态门控网络...")
        trainer.train_model(
            train_loader,
            val_loader,
            num_epochs=self.config['training']['num_epochs'],
            patience=self.config['training']['patience']
        )

        # 在测试集上预测
        test_loader = self._create_dataloader(self.X_test, self.y_test, shuffle=False)
        predictions, coefficients = trainer.predict(test_loader)

        # 反标准化预测结果
        predictions_original = self._inverse_transform_predictions(predictions)

        # 保存结果
        self.strategy_c_results = {
            'model': dynamic_model,
            'trainer': trainer,
            'predictions': predictions,
            'predictions_original': predictions_original,
            'coefficients': coefficients
        }

        # 保存模型和结果
        trainer.save_model(f"{self.results_dir}/dynamic_fusion_model.pth")
        np.save(f"{self.results_dir}/strategy_c_predictions_scaled.npy", predictions)
        np.save(f"{self.results_dir}/strategy_c_predictions_original.npy", predictions_original)
        np.save(f"{self.results_dir}/strategy_c_coefficients.npy", coefficients)

        logger.info("策略C完成")

        return self.strategy_c_results

    def calculate_metrics(self, predictions, true_values, is_scaled=True):
        """计算评估指标"""
        # 如果是标准化的数据，先反标准化
        if is_scaled:
            preds = self._inverse_transform_predictions(predictions)
            trues = self._inverse_transform_predictions(true_values)
        else:
            preds = predictions
            trues = true_values

        # 计算基本指标
        mae = mean_absolute_error(trues.flatten(), preds.flatten())
        rmse = np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
        r2 = r2_score(trues.flatten(), preds.flatten())

        # 计算MAPE（避免除以零）
        mask = trues != 0
        if np.any(mask):
            mape = mean_absolute_percentage_error(
                trues[mask].flatten(),
                preds[mask].flatten()
            ) * 100  # 转换为百分比
        else:
            mape = np.nan

        # 计算前5%最大波峰的误差
        peak_percentage = self.config['evaluation']['peak_percentage']
        flattened_trues = trues.flatten()
        flattened_preds = preds.flatten()

        # 找到峰值阈值
        peak_threshold = np.percentile(flattened_trues, 100 - peak_percentage * 100)
        peak_mask = flattened_trues >= peak_threshold

        if np.any(peak_mask):
            peak_rmse = np.sqrt(mean_squared_error(
                flattened_trues[peak_mask],
                flattened_preds[peak_mask]
            ))
            peak_mae = mean_absolute_error(
                flattened_trues[peak_mask],
                flattened_preds[peak_mask]
            )
        else:
            peak_rmse = np.nan
            peak_mae = np.nan

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'peak_rmse': peak_rmse,
            'peak_mae': peak_mae
        }

    def evaluate_all_strategies(self):
        """评估所有策略"""
        logger.info("评估所有策略...")

        # 准备真实值（反标准化）
        self.y_test_original = self._inverse_transform_predictions(self.y_test)

        # 评估专家模型
        patchtst_metrics = self.calculate_metrics(
            self.expert_predictions['patchtst'],
            self.y_test
        )

        nhits_metrics = self.calculate_metrics(
            self.expert_predictions['nhits'],
            self.y_test
        )

        # 评估融合策略
        strategy_a_metrics = self.calculate_metrics(
            self.strategy_a_results['predictions'],
            self.y_test
        )

        strategy_b_metrics = self.calculate_metrics(
            self.strategy_b_results['predictions'],
            self.y_test
        )

        strategy_c_metrics = self.calculate_metrics(
            self.strategy_c_results['predictions'],
            self.y_test
        )

        # 整理结果
        self.results = {
            'PatchTST': patchtst_metrics,
            'NHITS': nhits_metrics,
            '策略A (静态优化权重)': strategy_a_metrics,
            '策略B (广义线性融合)': strategy_b_metrics,
            '策略C (动态门控网络)': strategy_c_metrics
        }

        # 保存结果
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(f"{self.results_dir}/evaluation_results.csv")

        logger.info("评估完成，结果已保存")
        return self.results

    def visualize_results(self):
        """可视化结果"""
        logger.info("生成可视化结果...")

        # 1. 性能指标对比图
        self._plot_metrics_comparison()

        # 2. 预测示例图
        self._plot_prediction_examples()

        # 3. 策略C权重分析
        self._analyze_strategy_c_weights()

        logger.info("可视化完成")

    def _plot_metrics_comparison(self):
        """绘制指标对比图"""
        metrics = ['MAE', 'RMSE', 'MAPE', 'R2', 'peak_rmse']
        results_df = pd.DataFrame(self.results).T

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            results_df[metric].sort_values().plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title(f'{metric} 对比', fontsize=14)
            ax.set_xlabel(metric, fontsize=12)
            ax.grid(axis='x', linestyle='--', alpha=0.7)

            # 为R2特殊处理，范围设置在0-1
            if metric == 'R2':
                ax.set_xlim(0, 1)

        # 移除最后一个空图
        fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig(f"{self.visualization_dir}/metrics_comparison.png")
        plt.close()

    def _plot_prediction_examples(self, n_examples=3):
        """绘制预测示例"""
        # 随机选择几个样本
        indices = np.random.choice(len(self.y_test), n_examples, replace=False)

        for idx in indices:
            fig, ax = plt.subplots(figsize=(12, 6))

            # 真实值
            ax.plot(self.y_test_original[idx], label='真实值', color='black', linewidth=2)

            # 专家模型预测
            ax.plot(self.expert_predictions_original['patchtst'][idx], label='PatchTST', linestyle='--', alpha=0.8)
            ax.plot(self.expert_predictions_original['nhits'][idx], label='NHITS', linestyle='--', alpha=0.8)

            # 融合策略预测
            ax.plot(self.strategy_a_results['predictions_original'][idx], label='策略A', alpha=0.8)
            ax.plot(self.strategy_b_results['predictions_original'][idx], label='策略B', alpha=0.8)
            ax.plot(self.strategy_c_results['predictions_original'][idx], label='策略C', alpha=0.8)

            ax.set_title(f'预测示例 {idx}', fontsize=14)
            ax.set_xlabel('时间步', fontsize=12)
            ax.set_ylabel('波浪高度 (m)', fontsize=12)
            ax.legend()
            ax.grid(linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(f"{self.visualization_dir}/prediction_example_{idx}.png")
            plt.close()

    def _analyze_strategy_c_weights(self):
        """分析策略C的权重"""
        coefficients = self.strategy_c_results['coefficients']  # [n_samples, horizon, 3]
        w0 = coefficients[:, :, 0]  # 截距项
        w1 = coefficients[:, :, 1]  # PatchTST权重
        w2 = coefficients[:, :, 2]  # NHITS权重

        # 1. 绘制权重分布
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(w0.flatten(), bins=50, alpha=0.7)
        axes[0].set_title('w0 (截距项) 分布', fontsize=14)
        axes[0].grid(linestyle='--', alpha=0.7)

        axes[1].hist(w1.flatten(), bins=50, alpha=0.7)
        axes[1].set_title('w1 (PatchTST系数) 分布', fontsize=14)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].axvline(x=1, color='g', linestyle='--')
        axes[1].grid(linestyle='--', alpha=0.7)

        axes[2].hist(w2.flatten(), bins=50, alpha=0.7)
        axes[2].set_title('w2 (NHITS系数) 分布', fontsize=14)
        axes[2].axvline(x=0, color='r', linestyle='--')
        axes[2].axvline(x=1, color='g', linestyle='--')
        axes[2].grid(linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{self.visualization_dir}/strategy_c_weights_distribution.png")
        plt.close()

        # 2. 寻找权重超出[0,1]范围的时刻
        w1_outside = np.where((w1 < 0) | (w1 > 1))
        w2_outside = np.where((w2 < 0) | (w2 > 1))

        logger.info(f"策略C中，w1有 {len(w1_outside[0])} 个时刻超出[0,1]范围")
        logger.info(f"策略C中，w2有 {len(w2_outside[0])} 个时刻超出[0,1]范围")

        # 3. 绘制几个权重变化的例子
        n_examples = 3
        indices = np.random.choice(coefficients.shape[0], n_examples, replace=False)

        for idx in indices:
            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(w0[idx], label='w0 (截距项)', alpha=0.8)
            ax.plot(w1[idx], label='w1 (PatchTST)', alpha=0.8)
            ax.plot(w2[idx], label='w2 (NHITS)', alpha=0.8)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=1, color='g', linestyle='--', alpha=0.5)

            ax.set_title(f'动态系数示例 {idx}', fontsize=14)
            ax.set_xlabel('时间步', fontsize=12)
            ax.set_ylabel('系数值', fontsize=12)
            ax.legend()
            ax.grid(linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(f"{self.visualization_dir}/dynamic_coefficients_example_{idx}.png")
            plt.close()

    def implement_strategy_a_quick(self, max_iterations: int = 20):
        """实现策略A的快速模式：静态优化权重（减少MPA迭代次数）"""
        logger.info(f"实现策略A快速模式：静态优化权重（max_iterations={max_iterations}）...")

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

        # 使用MPA优化权重（快速模式，减少迭代次数）
        mpa_config = self.config['mpa'].copy()
        mpa_config['max_iterations'] = max_iterations  # 使用快速模式的迭代次数

        optimizer = MPAOptimizer.StaticWeightOptimizer(mpa_config)

        optimal_weights, best_score = optimizer.mpa.optimize_weights(
            val_expert_preds_combined, self.y_val
        )

        # 在测试集上应用权重
        strategy_a_predictions = np.zeros_like(self.y_test)
        for i, weight in enumerate(optimal_weights):
            strategy_a_predictions += weight * expert_preds_combined[:, :, i]

        # 反标准化预测结果
        strategy_a_predictions_original = self._inverse_transform_predictions(strategy_a_predictions)

        # 保存结果
        self.strategy_a_results = {
            'predictions': strategy_a_predictions,
            'predictions_original': strategy_a_predictions_original,
            'weights': optimal_weights,
            'validation_score': best_score
        }

        np.save(f"{self.results_dir}/strategy_a_predictions_scaled.npy", strategy_a_predictions)
        np.save(f"{self.results_dir}/strategy_a_predictions_original.npy", strategy_a_predictions_original)
        np.save(f"{self.results_dir}/strategy_a_weights.npy", optimal_weights)

        logger.info(f"策略A快速模式完成 - 最优权重: {optimal_weights}, 验证集RMSE: {best_score:.6f}")

        return self.strategy_a_results

    def implement_strategy_b_quick(self, max_iterations: int = 20):
        """实现策略B的快速模式：广义线性融合（减少MPA迭代次数）"""
        logger.info(f"实现策略B快速模式：广义线性融合（max_iterations={max_iterations}）...")

        # 准备数据
        patchtst_pred = self.expert_predictions['patchtst']
        nhits_pred = self.expert_predictions['nhits']

        # 合并预测 [n_samples, horizon, n_experts]
        expert_preds_combined = np.stack([patchtst_pred, nhits_pred], axis=2)

        # 使用验证集来优化系数
        val_loader = self._create_dataloader(self.X_val, self.y_val, shuffle=False)

        val_predictions = {}
        for name, expert_data in self.expert_models.items():
            trainer = expert_data['trainer']
            predictions = trainer.predict(val_loader)
            val_predictions[name] = predictions

        val_expert_preds_combined = np.stack([val_predictions['patchtst'], val_predictions['nhits']], axis=2)

        # 使用MPA优化系数（快速模式，减少迭代次数）
        mpa_config = self.config['mpa'].copy()
        mpa_config['max_iterations'] = max_iterations  # 使用快速模式的迭代次数

        optimizer = StackingOptimizer(mpa_config)

        optimal_coefficients, best_score = optimizer.optimize_coefficients(
            val_expert_preds_combined, self.y_val
        )

        # 在测试集上应用系数
        # 系数包括截距项和权重
        w0 = optimal_coefficients[0]
        weights = optimal_coefficients[1:]

        strategy_b_predictions = np.full_like(self.y_test, w0)
        for i, weight in enumerate(weights):
            strategy_b_predictions += weight * expert_preds_combined[:, :, i]

        # 反标准化预测结果
        strategy_b_predictions_original = self._inverse_transform_predictions(strategy_b_predictions)

        # 保存结果
        self.strategy_b_results = {
            'predictions': strategy_b_predictions,
            'predictions_original': strategy_b_predictions_original,
            'coefficients': optimal_coefficients,
            'validation_score': best_score
        }

        np.save(f"{self.results_dir}/strategy_b_predictions_scaled.npy", strategy_b_predictions)
        np.save(f"{self.results_dir}/strategy_b_predictions_original.npy", strategy_b_predictions_original)
        np.save(f"{self.results_dir}/strategy_b_coefficients.npy", optimal_coefficients)

        logger.info(f"策略B快速模式完成 - 最优系数: {optimal_coefficients}, 验证集RMSE: {best_score:.6f}")

        return self.strategy_b_results








    def run_complete_experiment(self, optimize_hyperparameters: bool = True):
        """运行完整实验流程"""
        logger.info("开始完整实验流程...")

        # 1. 数据加载和预处理
        self.load_and_preprocess_data()

        # 2. 超参数优化（可选）
        if optimize_hyperparameters:
            self.optimize_expert_models()

        # 3. 训练专家模型
        self.train_expert_models(use_optimized_params=optimize_hyperparameters)

        # 4. 获取专家模型预测
        self.get_expert_predictions()

        # 5. 执行融合策略
        self.implement_strategy_a()
        self.implement_strategy_b()
        self.implement_strategy_c()

        # 6. 评估所有策略
        self.evaluate_all_strategies()

        # 7. 可视化结果
        self.visualize_results()

        # 8. 保存最终结果表格
        results_df = pd.DataFrame(self.results).T
        print("\n" + "=" * 80)
        print("实验结果汇总")
        print("=" * 80)
        print(results_df.round(4))
        print("\n" + "=" * 80)

        logger.info("完整实验流程完成")
        return self.results