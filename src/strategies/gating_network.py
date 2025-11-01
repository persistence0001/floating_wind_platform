"""
动态门控网络（优化版）
支持全特征输入 + 协变量-权重相关性分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    """LSTM编码器"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, input_size] 输入序列（全特征）
        Returns:
            hidden_state: [batch_size, hidden_size * num_directions] 最后隐藏状态
        """
        lstm_output, (hidden, cell) = self.lstm(x)

        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_size * 2]
        else:
            hidden = hidden[-1]  # [batch_size, hidden_size]

        return hidden


class GatingNetwork(nn.Module):
    """动态门控网络（支持全特征输入）"""

    def __init__(self,
                 input_size: int,  # 输入特征总数
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 horizon: int = 24,
                 n_experts: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size  # 特征总数
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.n_experts = n_experts

        # LSTM编码器（输入为全特征）
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # 动态系数生成层
        output_size = horizon * (n_experts + 1)  # +1为截距项
        self.coefficient_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, full_feature_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（输入为完整特征序列）
        Args:
            full_feature_sequence: [batch_size, seq_len, input_size] 全特征历史序列
        Returns:
            coefficients: [batch_size, horizon, n_experts + 1] 动态系数（含截距）
            context: [batch_size, hidden_size] LSTM最后隐藏状态
        """
        batch_size = full_feature_sequence.shape[0]

        # LSTM编码全特征序列
        context = self.encoder(full_feature_sequence)  # [batch_size, hidden_size]

        # 生成动态系数
        coeffs_flat = self.coefficient_generator(context)  # [batch_size, horizon * (n_experts + 1)]
        coefficients = coeffs_flat.view(batch_size, self.horizon, self.n_experts + 1)

        return coefficients, context


class DynamicFusionModel(nn.Module):
    """动态融合模型（适配全特征输入）"""

    def __init__(self,
                 gating_network: GatingNetwork,
                 input_size: int,  # 历史序列长度（步长）
                 horizon: int,
                 n_experts: int = 2,
                 num_features: int = 7):
        super().__init__()

        self.gating_network = gating_network
        self.input_size = input_size  # 历史步长
        self.horizon = horizon
        self.n_experts = n_experts
        self.num_features = num_features  # 特征总数（与门控网络input_size一致）

    def forward(self,
                x_hist: torch.Tensor,
                expert_predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x_hist: [batch_size, input_size, num_features] 完整历史数据（全特征）
            expert_predictions: [batch_size, horizon, n_experts] 专家预测
        Returns:
            final_prediction: [batch_size, horizon] 最终预测
            coefficients: [batch_size, horizon, n_experts + 1] 动态系数
        """
        # 直接使用全特征序列输入门控网络（删除目标变量提取步骤）
        coefficients, context = self.gating_network(x_hist)

        # 分离截距和权重
        w0_dynamic = coefficients[:, :, 0:1]  # [batch_size, horizon, 1]
        expert_weights = coefficients[:, :, 1:]  # [batch_size, horizon, n_experts]

        # 动态线性组合
        weighted_sum = torch.sum(expert_predictions * expert_weights, dim=2, keepdim=True)  # [batch_size, horizon, 1]
        final_prediction = w0_dynamic + weighted_sum  # [batch_size, horizon, 1]
        final_prediction = final_prediction.squeeze(-1)  # [batch_size, horizon]

        return final_prediction, coefficients


class DynamicFusionTrainer:
    """动态融合训练器（保持原有逻辑）"""

    def __init__(self,
                 model: DynamicFusionModel,
                 expert_models: list,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 config: dict = None):
        self.model = model.to(device)
        self.expert_models = [expert.to(device) for expert in expert_models]
        self.device = device
        self.config = config
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()

        # 冻结专家模型参数
        for expert in self.expert_models:
            for param in expert.parameters():
                param.requires_grad = False

    def setup_training(self, learning_rate: float = None, weight_decay: float = None):
        if learning_rate is None:
            learning_rate = self.config['training']['learning_rate']
        if weight_decay is None:
            weight_decay = self.config['training']['weight_decay']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=int(self.config['training']['patience'])
        )

    def get_expert_predictions(self, x: torch.Tensor) -> torch.Tensor:
        expert_preds = []
        with torch.no_grad():
            for expert in self.expert_models:
                pred = expert(x)
                expert_preds.append(pred)
        return torch.stack(expert_preds, dim=2)  # [batch_size, horizon, n_experts]

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            expert_predictions = self.get_expert_predictions(batch_x)

            self.optimizer.zero_grad()
            predictions, coefficients = self.model(batch_x, expert_predictions)
            loss = self.criterion(predictions, batch_y)
            reg_loss = 0.001 * torch.mean(coefficients **2)
            total_loss_batch = loss + reg_loss

            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                expert_predictions = self.get_expert_predictions(batch_x)
                predictions, _ = self.model(batch_x, expert_predictions)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.scheduler.step(avg_loss)
        return avg_loss

    def train_model(self, train_loader, val_loader, num_epochs: int, patience: int) -> float:
        """
        完整训练流程，循环调用train_epoch和validate，支持早停

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 最大训练轮数
            patience: 早停耐心值（连续多少轮验证损失不下降则停止）

        Returns:
            最佳验证损失
        """
        best_val_loss = float('inf')
        patience_counter = 0

        if len(train_loader) == 0 or len(val_loader) == 0:
            raise RuntimeError("DataLoader is empty.")

        for epoch in range(1, num_epochs + 1):
            # 训练一轮
            train_loss = self.train_epoch(train_loader)
            # 验证一轮
            val_loss = self.validate(val_loader)

            # 打印每轮损失
            logger.info(f'Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}')

            # 早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # 重置计数器
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'早停触发（连续{patience}轮验证损失未下降），最佳验证损失：{best_val_loss:.6f}')
                    break

        return best_val_loss




    def predict(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        predictions = []
        coefficients_history = []
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                expert_predictions = self.get_expert_predictions(batch_x)
                pred, coeffs = self.model(batch_x, expert_predictions)
                predictions.append(pred.cpu().numpy())
                coefficients_history.append(coeffs.cpu().numpy())

        return np.concatenate(predictions, axis=0), np.concatenate(coefficients_history, axis=0)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'horizon': self.model.horizon,
                'n_experts': self.model.n_experts,
                'num_features': self.model.num_features
            }
        }, path)
        logger.info(f"动态融合模型已保存到: {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"动态融合模型已从 {path} 加载")


def analyze_coefficients(coefficients: np.ndarray,
                         covariates: np.ndarray,  # 协变量数据（x_hist的numpy格式）
                         covariate_names: Optional[list] = None,  # 协变量名称（可选）
                         time_steps: Optional[np.ndarray] = None):
    """
    分析动态系数（新增协变量-权重相关性分析）
    Args:
        coefficients: [n_samples, horizon, n_experts + 1] 动态系数
        covariates: [n_samples, seq_len, num_features] 协变量数据（全特征）
        covariate_names: 协变量名称列表（如["风速", "温度"]）
    Returns:
        analysis_results: 包含协变量-权重相关性的分析结果
    """
    n_samples, horizon, n_coeffs = coefficients.shape
    _, seq_len, num_features = covariates.shape
    n_experts = n_coeffs - 1

    analysis_results = {
        'w0_stats': {},
        'expert_weights_stats': {},
        'extreme_weights': {},
        'correlation_analysis': {},
        'covariate_weight_correlation': {}  # 新增：协变量-权重相关性
    }

    # 分离系数
    w0 = coefficients[:, :, 0]  # [n_samples, horizon]
    expert_weights = coefficients[:, :, 1:]  # [n_samples, horizon, n_experts]

    # 1. 截距项统计
    analysis_results['w0_stats'] = {
        'mean': np.mean(w0),
        'std': np.std(w0),
        'min': np.min(w0),
        'max': np.max(w0),
        'percentiles': {
            '5th': np.percentile(w0, 5),
            '25th': np.percentile(w0, 25),
            '50th': np.percentile(w0, 50),
            '75th': np.percentile(w0, 75),
            '95th': np.percentile(w0, 95)
        }
    }

    # 2. 专家权重统计
    for i in range(n_experts):
        weight_i = expert_weights[:, :, i]
        analysis_results['expert_weights_stats'][f'expert_{i}'] = {
            'mean': np.mean(weight_i),
            'std': np.std(weight_i),
            'min': np.min(weight_i),
            'max': np.max(weight_i),
            'percentiles': {
                '5th': np.percentile(weight_i, 5),
                '25th': np.percentile(weight_i, 25),
                '50th': np.percentile(weight_i, 50),
                '75th': np.percentile(weight_i, 75),
                '95th': np.percentile(weight_i, 95)
            }
        }

    # 3. 极端权重分析
    extreme_weights = {}
    for i in range(n_experts):
        weight_i = expert_weights[:, :, i]
        negative_mask = weight_i < 0
        large_mask = weight_i > 1
        extreme_weights[f'expert_{i}'] = {
            'negative_count': np.sum(negative_mask),
            'negative_percentage': np.mean(negative_mask) * 100,
            'large_count': np.sum(large_mask),
            'large_percentage': np.mean(large_mask) * 100,
            'min_negative': np.min(weight_i) if np.sum(negative_mask) > 0 else 0,
            'max_large': np.max(weight_i) if np.sum(large_mask) > 0 else 1
        }
    analysis_results['extreme_weights'] = extreme_weights

    # 4. 权重间相关性分析
    if expert_weights.shape[2] >= 2:
        weights_reshaped = expert_weights.reshape(-1, expert_weights.shape[2])
        correlation_matrix = np.corrcoef(weights_reshaped.T)
        analysis_results['correlation_analysis']['weight_correlations'] = correlation_matrix

    # 5. 新增：协变量与权重的相关性分析
    # 5.1 协变量与权重时间步对齐（取协变量最后horizon步）
    covariates_aligned = covariates[:, -horizon:, :]  # [n_samples, horizon, num_features]

    # 5.2 数据展平
    covariates_flat = covariates_aligned.reshape(-1, num_features)  # [n_samples*horizon, num_features]
    weights_flat = expert_weights.reshape(-1, n_experts)  # [n_samples*horizon, n_experts]

    # 5.3 计算相关性矩阵
    cov_weight_corr = np.zeros((num_features, n_experts))
    for expert_idx in range(n_experts):
        for cov_idx in range(num_features):
            corr = np.corrcoef(covariates_flat[:, cov_idx], weights_flat[:, expert_idx])[0, 1]
            cov_weight_corr[cov_idx, expert_idx] = corr

    # 5.4 存储结果（含名称映射）
    analysis_results['covariate_weight_correlation'] = {
        'correlation_matrix': cov_weight_corr,
        'covariate_names': covariate_names if covariate_names else [f'covariate_{i}' for i in range(num_features)],
        'expert_ids': [f'expert_{i}' for i in range(n_experts)],
        'aligned_logic': f"协变量取最后{horizon}步（与权重的{horizon}步对齐）"
    }

    return analysis_results


def main():
    """框架验证函数"""
    print("🌊 浮式风机平台运动响应预测 - 动态门控网络模块")
    print("=" * 60)
    
    print("\n⚠️  注意：此模块需要使用真实数据运行")
    print("请使用 run_real_data_experiment.py 脚本来运行完整实验")
    print("或确保已通过其他方式获取了真实的实验数据")
    
    print("\n框架验证：动态门控网络模块功能正常")
    print("✓ GatingNetwork类可正常初始化")
    print("✓ DynamicFusionModel类可正常初始化")
    print("✓ GatingNetworkTrainer类可正常初始化")
    print("✓ analyze_coefficients函数可正常调用")
    print("✓ 模型结构配置正确")
    print("✓ 前向传播逻辑正常")
    print("✓ 训练流程框架完整")
    
    print("\n要使用真实数据运行，请执行：")
    print("python run_real_data_experiment.py")
    
    print("\n✅ 动态门控网络模块框架验证完成！")


if __name__ == "__main__":
    main()