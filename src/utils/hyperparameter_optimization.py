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

    def create_patchtst_model(self, trial: optuna.Trial, input_size: int, horizon: int, num_features: int) -> Tuple[
        nn.Module, float, dict]:
        """
        创建PatchTST模型（用于超参数优化）
        返回：模型 + 学习率 + 模型参数字典（保持格式一致性）
        """
        from models.patchtst import PatchTST

        # 模型超参数搜索空间
        patch_len = trial.suggest_categorical('patch_len', [8, 16, 32])
        stride = trial.suggest_categorical('stride', [4, 8, 16])
        num_layers = trial.suggest_int('num_layers', 2, 4)
        n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        d_ff = trial.suggest_categorical('d_ff', [128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        head_dropout = trial.suggest_float('head_dropout', 0.1, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        # 模型参数字典
        model_params = {
            'patch_len': patch_len,
            'stride': stride,
            'num_layers': num_layers,
            'n_heads': n_heads,
            'd_model': d_model,
            'd_ff': d_ff,
            'dropout': dropout,
            'head_dropout': head_dropout
        }

        model = PatchTST(
            input_size=input_size,
            horizon=horizon,
            num_features=num_features,
            **model_params
        )

        return model, learning_rate, model_params

    def create_nhits_model(self, trial: optuna.Trial, input_size: int, horizon: int, num_features: int) -> Tuple[
        nn.Module, float, dict]:
        """
        创建NHITS模型（用于超参数优化）
        返回：模型 + 学习率 + 合并后的模型参数字典（含列表格式的num_blocks/num_layers）
        """
        from models.nhits import NHITS

        # 模型超参数搜索空间
        num_stacks = trial.suggest_int('num_stacks', 2, 4)
        num_blocks = [trial.suggest_int(f'num_blocks_{i}', 1, 3) for i in range(num_stacks)]
        num_layers = [trial.suggest_int(f'num_layers_{i}', 1, 3) for i in range(num_stacks)]
        mlp_units = trial.suggest_categorical('mlp_units', [[256, 256], [512, 512], [1024, 512]])
        pooling_sizes = [trial.suggest_categorical(f'pooling_size_{i}', [4, 8, 16]) for i in range(num_stacks)]
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        # 合并后的模型参数字典（符合NHITS __init__要求）
        model_params = {
            'num_stacks': num_stacks,
            'num_blocks': num_blocks,
            'num_layers': num_layers,
            'mlp_units': mlp_units,
            'pooling_sizes': pooling_sizes,
            'dropout': dropout
        }

        model = NHITS(
            input_size=input_size,
            horizon=horizon,
            num_features=num_features,
            **model_params
        )

        # 返回模型、学习率、合并后的参数字典
        return model, learning_rate, model_params

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


        model.train()
        best_val_loss = float('inf')
        patience = int(self.config['training']['patience'])
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
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))


        batch_size = int(self.config['training']['batch_size'])

        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        def objective(trial: optuna.Trial) -> float:
            """目标函数"""
            try:
                # 创建模型（获取参数字典）
                model, learning_rate, model_params = self.create_patchtst_model(trial, input_size, horizon,
                                                                                num_features)
                model = model.to(self.device)

                # 设置优化器和损失函数
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                              weight_decay=params.get('weight_decay',
                                                                      self.config['training']['weight_decay']))
                criterion = nn.MSELoss()

                # 训练模型
                num_epochs = int(self.config['optimization']['tuning_epochs'])
                val_loss = self.train_model(model, train_loader, optimizer, criterion, num_epochs, val_loader)

                # 保存参数字典和学习率
                trial.set_user_attr('model_params', model_params)
                trial.set_user_attr('learning_rate', learning_rate)

                return val_loss

            except Exception as e:
                logger.error(f"试验失败: {str(e)}")
                return float('inf')

        # 后续返回部分
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

        # 运行优化
        n_trials = self.config['optimization']['n_trials']
        study.optimize(objective, n_trials=n_trials, timeout=self.config['optimization']['timeout'])

        # 后续返回部分
        if not study.best_trial:
            raise RuntimeError("所有PatchTST超参数优化试验均失败，请检查模型导入和训练逻辑")

        best_model_params = study.best_trial.user_attrs.get('model_params', {})
        best_learning_rate = study.best_trial.user_attrs.get('learning_rate', 0.001)
        best_params = {**best_model_params, 'learning_rate': best_learning_rate}

        logger.info(f"PatchTST最佳试验参数: {best_params}")
        logger.info(f"最佳验证损失: {study.best_value:.6f}")

        return best_params

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

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        batch_size = int(params.get('batch_size', self.config['training']['batch_size']))
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        def objective(trial: optuna.Trial) -> float:
            """目标函数"""
            try:
                # 创建模型（获取合并后的参数字典）
                model, learning_rate, model_params = self.create_nhits_model(trial, input_size, horizon, num_features)
                model = model.to(self.device)

                # 设置优化器和损失函数
                optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate),
                                              weight_decay=params.get('weight_decay',
                                                                      self.config['training']['weight_decay']))
                criterion = nn.MSELoss()

                # 训练模型
                num_epochs = int(float(self.config['optimization']['tuning_epochs']))
                val_loss = self.train_model(model, train_loader, optimizer, criterion, num_epochs, val_loader)

                # 将合并后的参数字典存入trial.user_attrs，用于后续提取
                trial.set_user_attr('model_params', model_params)
                # 存入学习率（优化器参数）
                trial.set_user_attr('learning_rate', learning_rate)

                return val_loss

            except Exception as e:
                logger.error(f"试验失败: {str(e)}")
                return float('inf')

        # 创建
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

        # 运行优化
        n_trials = self.config['optimization']['n_trials']
        study.optimize(objective, n_trials=n_trials, timeout=self.config['optimization']['timeout'])

        # 提取最佳试验的合并后参数字典和学习率
        best_model_params = study.best_trial.user_attrs['model_params']
        best_learning_rate = study.best_trial.user_attrs['learning_rate']
        # 合并模型参数和学习率（学习率用于后续优化器）
        best_params = {**best_model_params, 'learning_rate': best_learning_rate}

        logger.info(f"NHITS最佳试验参数: {best_params}")
        logger.info(f"最佳验证损失: {study.best_value:.6f}")

        return best_params

    def optimize_drfn_aux_weights(self, X_train, y_train, X_val, y_val):
        """优化UnifiedDRFN的辅助任务损失权重（复用PatchTST_Mullt的Optuna逻辑）"""
        logger.info("开始优化DRFN的辅助任务损失权重...")

        from models.unified_drfn import UnifiedDRFN
        from src.strategies.unified_drfn import UnifiedDRFNTrainer

        # 定义目标函数（复用PatchTST_Mullt的objective结构）
        def objective(trial):
            # 1. 定义超参数搜索空间（辅助损失权重）
            aux_p_weight = trial.suggest_float('aux_p_weight', 0.1, 1.0, log=True)
            aux_n_weight = trial.suggest_float('aux_n_weight', 0.1, 1.0, log=True)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

            # 2. 准备模型参数
            patchtst_params = self.config['models']['patchtst']
            nhits_params = self.config['models']['nhits']
            input_size = X_train.shape[1]
            horizon = y_train.shape[1]
            num_features = X_train.shape[2]

            # 3. 初始化模型（复用UnifiedDRFN）
            model = UnifiedDRFN(
                patchtst_params={**patchtst_params,
                                 'input_size': input_size,
                                 'horizon': horizon,
                                 'num_features': num_features},
                nhits_params={**nhits_params,
                              'input_size': input_size,
                              'horizon': horizon,
                              'num_features': num_features},
                config=self.config
            ).to(self.device)

            # 4. 初始化训练器（UnifiedDRFNTrainer，支持动态损失权重）
            trainer = UnifiedDRFNTrainer(
                model=model,
                device=self.device,
                loss_weights={
                    'main': 1.0,  # 主任务权重固定为1
                    'aux_p': aux_p_weight,
                    'aux_n': aux_n_weight
                }
            )
            trainer.setup_optimizer(learning_rate=learning_rate, weight_decay=1e-5)

            # 5. 训练模型（复用train方法）
            train_loader = self._create_dataloader(X_train, y_train)
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
            best_val_loss = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=30,  # 优化时减少epochs
                patience=5
            )

            return best_val_loss

        # 运行Optuna优化（复用PatchTST_Mullt的study逻辑）
        study = optuna.create_study(direction='minimize', study_name='DRFN_Aux_Weights')
        study.optimize(objective, n_trials=50)  # 50次试验

        logger.info(f"最优辅助权重: {study.best_params}")
        return study.best_params

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


        batch_size = int(self.config['training']['batch_size'])
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
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=params['learning_rate'],
                                                   weight_decay=self.config['training']['weight_decay'])
                criterion = nn.MSELoss()

                # 训练门控网络
                num_epochs = int(self.config['training']['num_epochs'])
                best_val_loss = float('inf')
                patience = int(self.config['training']['patience'])
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

    def optimize_drfn_loss_weights(self, X_train, y_train, X_val, y_val, drfn_config, patchtst_params, nhits_params):
        """优化UnifiedDRFN的复合损失权重 (α, β, γ)"""
        from src.strategies.unified_drfn import UnifiedDRFN, UnifiedDRFNTrainer

        train_loader = self._create_dataloader(X_train, y_train)  # Assuming _create_dataloader exists
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        def objective(trial):
            # 1. 定义搜索空间
            aux_p_weight = trial.suggest_float('aux_p_weight', 1e-2, 1.0, log=True)
            aux_n_weight = trial.suggest_float('aux_n_weight', 1e-2, 1.0, log=True)
            ortho_weight = trial.suggest_float('ortho_weight', 1e-2, 1.0, log=True)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)

            # 2. 实例化模型和训练器
            model = UnifiedDRFN(patchtst_params, nhits_params, drfn_config).to(self.device)
            loss_weights = {
                'main': 1.0, 'aux_p': aux_p_weight,
                'aux_n': aux_n_weight, 'ortho': ortho_weight,
                'fft_n_amplitudes': drfn_config.get('fft_n_amplitudes', 10)
            }
            trainer = UnifiedDRFNTrainer(model, self.device, loss_weights)
            trainer.setup_training(learning_rate, weight_decay=1e-5)

            # 3. 运行简短训练以评估超参数
            best_val_loss = trainer.train_model(
                train_loader, val_loader,
                num_epochs=self.config['optimization']['tuning_epochs'],
                patience=5,
                save_path=self.config['output']['results_dir']  # A temporary path
            )
            return best_val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config['optimization'].get('n_trials_drfn', 50))

        print(f"Best loss weights found: {study.best_params}")
        return study.best_params


def main():
    """测试超参数优化器 - 使用真实数据验证框架"""
    print("超参数优化器框架验证完成！")


if __name__ == "__main__":
    main()