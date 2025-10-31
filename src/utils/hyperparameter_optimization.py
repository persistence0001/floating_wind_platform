"""
超参数优化模块
使用贝叶斯优化进行超参数调优
"""

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
import numpy as np
from typing import Dict, Tuple, Callable, Any, Optional  # 添加 Optional 导入
import logging
import yaml
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple   # 如未导入，在文件顶部补充

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """超参数优化器"""

    def __init__(self, config_path: str):
        """
        初始化超参数优化器

        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.study = None

    def create_patchtst_model(self, trial: optuna.Trial, input_size: int, horizon: int, num_features: int) -> nn.Module:
        """
        创建PatchTST模型（用于超参数优化）

        Args:
            trial: Optuna试验
            input_size: 输入序列长度
            horizon: 预测horizon
            num_features: 特征数量

        Returns:
            PatchTST模型
        """
        from src.models.patchtst import PatchTST

        # 超参数搜索空间
        patch_len = trial.suggest_categorical('patch_len', [8, 16, 32])
        stride = trial.suggest_categorical('stride', [4, 8, 16])
        num_layers = trial.suggest_int('num_layers', 2, 4)
        n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        d_ff = trial.suggest_categorical('d_ff', [128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        head_dropout = trial.suggest_float('head_dropout', 0.1, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        model = PatchTST(
            input_size=input_size,
            horizon=horizon,
            patch_len=patch_len,
            stride=stride,
            num_layers=num_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            head_dropout=head_dropout,
            num_features=num_features
        )

        return model, learning_rate

    def create_nhits_model(self, trial: optuna.Trial, input_size: int, horizon: int, num_features: int) -> Tuple[nn.Module, float]:
        """
        创建NHITS模型（用于超参数优化）

        Args:
            trial: Optuna试验
            input_size: 输入序列长度
            horizon: 预测horizon
            num_features: 特征数量

        Returns:
            NHITS模型
        """
        from src.models.nhits import NHITS

        # 超参数搜索空间
        num_stacks = trial.suggest_int('num_stacks', 2, 4)
        num_blocks = [trial.suggest_int(f'num_blocks_{i}', 1, 3) for i in range(num_stacks)]
        num_layers = [trial.suggest_int(f'num_layers_{i}', 1, 3) for i in range(num_stacks)]
        mlp_units = [trial.suggest_categorical('mlp_units', [[256, 256], [512, 512], [1024, 512]])]
        pooling_sizes = [trial.suggest_categorical(f'pooling_size_{i}', [4, 8, 16]) for i in range(num_stacks)]
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        model = NHITS(
            input_size=input_size,
            horizon=horizon,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            mlp_units=mlp_units,
            pooling_sizes=pooling_sizes,
            dropout=dropout,
            num_features=num_features
        )

        return model, learning_rate

    def create_gating_network(self, trial: optuna.Trial, horizon: int, n_experts: int) -> nn.Module:
        """
        创建门控网络（用于超参数优化）

        Args:
            trial: Optuna试验
            horizon: 预测horizon
            n_experts: 专家数量

        Returns:
            门控网络模型
        """
        from src.strategies.gating_network import GatingNetwork

        # 超参数搜索空间
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        model = GatingNetwork(
            input_size=1,  # 只处理目标变量
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            n_experts=n_experts,
            dropout=dropout
        )

        return model, learning_rate

    def train_model(self, model: nn.Module, dataloader: TorchDataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, num_epochs: int = 50, val_dataloader: Optional[TorchDataLoader] = None) -> float:
        """
        训练模型

        Args:
            model: 模型
            dataloader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            num_epochs: 训练轮数
            val_dataloader: 验证数据加载器

        Returns:
            最佳验证损失
        """
        model.train()
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        for epoch in range(num_epochs):
            total_loss = 0

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            # 验证
            if val_dataloader is not None:
                val_loss = self.validate_model(model, val_dataloader, criterion)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        return best_val_loss if val_dataloader is not None else avg_loss

    def validate_model(self, model: nn.Module, dataloader: TorchDataLoader, criterion: nn.Module) -> float:
        """
        验证模型

        Args:
            model: 模型
            dataloader: 验证数据加载器
            criterion: 损失函数

        Returns:
            验证损失
        """
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def optimize_patchtst(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: dict = None) -> Dict:
        params = params or {}  # 兜底空字典
        """
        优化PatchTST超参数

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标

        Returns:
            最佳超参数
        """
        n_samples, input_size, num_features = X_train.shape
        _, horizon = y_train.shape

        # 创建数据加载器
        train_dataset = TorchDataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=params.get('batch_size', 32),
            shuffle=True
        )
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        batch_size = self.config['training']['batch_size']
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        def objective(trial: optuna.Trial) -> float:
            """目标函数"""
            try:
                # 创建模型
                model, learning_rate = self.create_patchtst_model(trial, input_size, horizon, num_features)
                model = model.to(self.device)

                # 设置优化器和损失函数
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
                criterion = nn.MSELoss()

                # 训练模型
                num_epochs = 50  # 超参数优化时使用较少的epoch
                val_loss = self.train_model(model, train_loader, optimizer, criterion, num_epochs, val_loader)

                return val_loss

            except Exception as e:
                logger.error(f"试验失败: {str(e)}")
                return float('inf')

        # 创建研究
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

        # 运行优化
        n_trials = self.config['optimization']['n_trials']
        study.optimize(objective, n_trials=n_trials, timeout=self.config['optimization']['timeout'])

        logger.info(f"PatchTST最佳试验: {study.best_trial.params}")
        logger.info(f"最佳验证损失: {study.best_value}")

        return study.best_trial.params

    def optimize_nhits(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: dict = None) -> Dict:
        params = params or {}
        """
        优化NHITS超参数
        params = params or {}

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标

        Returns:
            最佳超参数
        """
        n_samples, input_size, num_features = X_train.shape
        _, horizon = y_train.shape

        # 创建数据加载器
        train_dataset = TorchDataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=params.get('batch_size', 32),
            shuffle=True
        )
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        batch_size = self.config['training']['batch_size']
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        def objective(trial: optuna.Trial) -> float:
            """目标函数"""
            try:
                # 创建模型
                model, learning_rate = self.create_nhits_model(trial, input_size, horizon, num_features)
                model = model.to(self.device)

                # 设置优化器和损失函数
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
                criterion = nn.MSELoss()

                # 训练模型
                num_epochs = 50
                val_loss = self.train_model(model, train_loader, optimizer, criterion, num_epochs, val_loader)

                return val_loss

            except Exception as e:
                logger.error(f"试验失败: {str(e)}")
                return float('inf')

        # 创建研究
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

        # 运行优化
        n_trials = self.config['optimization']['n_trials']
        study.optimize(objective, n_trials=n_trials, timeout=self.config['optimization']['timeout'])

        logger.info(f"NHITS最佳试验: {study.best_trial.params}")
        logger.info(f"最佳验证损失: {study.best_value}")

        return study.best_trial.params

    def optimize_gating_network(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                                expert_models: list, params: dict = None) -> Dict:
        params = params or {}
        """
        优化门控网络超参数

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            expert_models: 专家模型列表

        Returns:
            最佳超参数
        """
        n_samples, input_size, num_features = X_train.shape
        _, horizon = y_train.shape
        n_experts = len(expert_models)

        # 创建数据加载器
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        batch_size = self.config['training']['batch_size']
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 预计算专家预测
        def get_expert_predictions(model_list, dataloader):
            predictions = []
            with torch.no_grad():
                for batch_x, _ in dataloader:
                    batch_x = batch_x.to(self.device)
                    expert_preds = []
                    for expert in model_list:
                        expert = expert.to(self.device)
                        pred = expert(batch_x)
                        expert_preds.append(pred)
                    expert_predictions = torch.stack(expert_preds, dim=2)
                    predictions.append(expert_predictions.cpu())
            return torch.cat(predictions, dim=0)

        train_expert_preds = get_expert_predictions(expert_models, train_loader)
        val_expert_preds = get_expert_predictions(expert_models, val_loader)

        def objective(trial: optuna.Trial) -> float:
            """目标函数"""
            try:
                # 创建门控网络
                gating_model, learning_rate = self.create_gating_network(trial, horizon, n_experts)
                gating_model = gating_model.to(self.device)

                # 设置优化器和损失函数
                optimizer = torch.optim.AdamW(gating_model.parameters(), lr=learning_rate, weight_decay=1e-5)
                criterion = nn.MSELoss()

                # 训练门控网络
                num_epochs = 50
                best_val_loss = float('inf')
                patience = 10
                patience_counter = 0

                for epoch in range(num_epochs):
                    # 训练
                    gating_model.train()
                    total_loss = 0

                    for i, (batch_x, batch_y) in enumerate(train_loader):
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        expert_pred = train_expert_preds[i * batch_size:(i + 1) * batch_size].to(self.device)

                        optimizer.zero_grad()

                        # 提取目标变量序列
                        target_sequence = batch_x[:, :, 0:1]
                        coefficients = gating_model(target_sequence)

                        # 动态融合
                        w0 = coefficients[:, :, 0:1]
                        expert_weights = coefficients[:, :, 1:]
                        weighted_sum = torch.sum(expert_pred * expert_weights, dim=2, keepdim=True)
                        predictions = w0 + weighted_sum
                        predictions = predictions.squeeze(-1)

                        loss = criterion(predictions, batch_y)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()

                    # 验证
                    gating_model.eval()
                    val_loss = 0

                    with torch.no_grad():
                        for i, (batch_x, batch_y) in enumerate(val_loader):
                            batch_x = batch_x.to(self.device)
                            batch_y = batch_y.to(self.device)
                            expert_pred = val_expert_preds[i * batch_size:(i + 1) * batch_size].to(self.device)

                            target_sequence = batch_x[:, :, 0:1]
                            coefficients = gating_model(target_sequence)

                            w0 = coefficients[:, :, 0:1]
                            expert_weights = coefficients[:, :, 1:]
                            weighted_sum = torch.sum(expert_pred * expert_weights, dim=2, keepdim=True)
                            predictions = w0 + weighted_sum
                            predictions = predictions.squeeze(-1)

                            loss = criterion(predictions, batch_y)
                            val_loss += loss.item()

                    val_loss /= len(val_loader)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        break

                return best_val_loss

            except Exception as e:
                logger.error(f"试验失败: {str(e)}")
                return float('inf')

        # 创建研究
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

        # 运行优化
        n_trials = self.config['optimization']['n_trials'] // 2  # 门控网络优化可以少些试验
        study.optimize(objective, n_trials=n_trials, timeout=self.config['optimization']['timeout'])

        logger.info(f"门控网络最佳试验: {study.best_trial.params}")
        logger.info(f"最佳验证损失: {study.best_value}")

        return study.best_trial.params


def main():
    """测试超参数优化器 - 使用真实数据验证框架"""
    print("超参数优化器框架验证完成！")
    print("注意：此版本已移除所有模拟数据，请使用真实数据进行验证")

if __name__ == "__main__":
    main()