"""
动态门控网络
为每个输入序列动态生成融合权重
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
            x: [batch_size, seq_len, input_size] 输入序列

        Returns:
            hidden_state: [batch_size, hidden_size * num_directions] 最后隐藏状态
        """
        lstm_output, (hidden, cell) = self.lstm(x)

        # 如果是双向LSTM，需要拼接两个方向的隐藏状态
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_size * 2]
        else:
            hidden = hidden[-1]  # [batch_size, hidden_size]

        return hidden


class GatingNetwork(nn.Module):
    """动态门控网络"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 horizon: int = 24,
                 n_experts: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size  # 目标变量的维度（应该是1）
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.n_experts = n_experts

        # LSTM编码器 - 只处理目标变量
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # 动态系数生成层
        # 输出维度: horizon * (n_experts + 1)
        # +1 是为了截距项 w0
        output_size = horizon * (n_experts + 1)

        self.coefficient_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
            # 注意：这里没有激活函数，允许任意实数输出
        )

    def forward(self, target_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            target_sequence: [batch_size, seq_len, 1] 目标变量历史序列

        Returns:
            coefficients: [batch_size, horizon, n_experts + 1] 动态系数
                          包含 [w0, w1, w2, ...] 其中w0是截距
        """
        batch_size = target_sequence.shape[0]

        # LSTM编码
        context = self.encoder(target_sequence)  # [batch_size, hidden_size]

        # 生成动态系数
        coeffs_flat = self.coefficient_generator(context)  # [batch_size, horizon * (n_experts + 1)]

        # 重塑为 [batch_size, horizon, n_experts + 1]
        coefficients = coeffs_flat.view(batch_size, self.horizon, self.n_experts + 1)

        return coefficients, context


class DynamicFusionModel(nn.Module):
    """动态融合模型"""

    def __init__(self,
                 gating_network: GatingNetwork,
                 input_size: int,
                 horizon: int,
                 n_experts: int = 2,
                 num_features: int = 7):
        super().__init__()

        self.gating_network = gating_network
        self.input_size = input_size
        self.horizon = horizon
        self.n_experts = n_experts
        self.num_features = num_features

    def forward(self,
                x_hist: torch.Tensor,
                expert_predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x_hist: [batch_size, input_size, num_features] 完整历史数据
            expert_predictions: [batch_size, horizon, n_experts] 专家预测

        Returns:
            final_prediction: [batch_size, horizon] 最终预测
            coefficients: [batch_size, horizon, n_experts + 1] 动态系数
        """
        # 提取目标变量序列 (假设第一列是目标变量)
        target_sequence = x_hist[:, :, 0:1]  # [batch_size, input_size, 1]

        # 门控网络生成动态系数
        coefficients, context = self.gating_network(target_sequence)
        # coefficients: [batch_size, horizon, n_experts + 1]

        # 分离截距和权重
        w0_dynamic = coefficients[:, :, 0:1]  # [batch_size, horizon, 1]
        expert_weights = coefficients[:, :, 1:]  # [batch_size, horizon, n_experts]

        # 动态线性组合
        # expert_predictions: [batch_size, horizon, n_experts]
        # expert_weights: [batch_size, horizon, n_experts]

        # 计算加权和
        weighted_sum = torch.sum(expert_predictions * expert_weights, dim=2, keepdim=True)  # [batch_size, horizon, 1]

        # 添加截距
        final_prediction = w0_dynamic + weighted_sum  # [batch_size, horizon, 1]

        # 移除最后一个维度
        final_prediction = final_prediction.squeeze(-1)  # [batch_size, horizon]

        return final_prediction, coefficients


class DynamicFusionTrainer:
    """动态融合训练器"""

    def __init__(self,
                 model: DynamicFusionModel,
                 expert_models: list,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.expert_models = [expert.to(device) for expert in expert_models]
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()

        # 冻结专家模型参数
        for expert in self.expert_models:
            for param in expert.parameters():
                param.requires_grad = False

    def setup_training(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        """设置训练参数"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

    def get_expert_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """获取专家模型预测"""
        expert_preds = []

        with torch.no_grad():
            for expert in self.expert_models:
                pred = expert(x)
                expert_preds.append(pred)

        # 堆叠预测结果 [batch_size, horizon, n_experts]
        expert_predictions = torch.stack(expert_preds, dim=2)

        return expert_predictions

    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 获取专家预测
            expert_predictions = self.get_expert_predictions(batch_x)

            self.optimizer.zero_grad()

            # 动态融合预测
            predictions, coefficients = self.model(batch_x, expert_predictions)
            loss = self.criterion(predictions, batch_y)

            # 添加正则化项防止权重过大
            reg_loss = 0.001 * torch.mean(coefficients ** 2)
            total_loss_batch = loss + reg_loss

            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 获取专家预测
                expert_predictions = self.get_expert_predictions(batch_x)

                # 动态融合预测
                predictions, _ = self.model(batch_x, expert_predictions)
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.scheduler.step(avg_loss)

        return avg_loss

    def predict(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        self.model.eval()
        predictions = []
        coefficients_history = []

        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)

                # 获取专家预测
                expert_predictions = self.get_expert_predictions(batch_x)

                # 动态融合预测
                pred, coeffs = self.model(batch_x, expert_predictions)
                predictions.append(pred.cpu().numpy())
                coefficients_history.append(coeffs.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        coefficients = np.concatenate(coefficients_history, axis=0)

        return predictions, coefficients

    def save_model(self, path: str):
        """保存模型"""
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
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"动态融合模型已从 {path} 加载")


def analyze_coefficients(coefficients: np.ndarray, time_steps: Optional[np.ndarray] = None):
    """
    分析动态系数

    Args:
        coefficients: [n_samples, horizon, n_experts + 1] 动态系数
        time_steps: 时间步（可选）

    Returns:
        analysis_results: 分析结果字典
    """
    n_samples, horizon, n_coeffs = coefficients.shape

    analysis_results = {
        'w0_stats': {},  # 截距项统计
        'expert_weights_stats': {},  # 专家权重统计
        'extreme_weights': {},  # 极端权重分析
        'correlation_analysis': {}  # 相关性分析
    }

    # 分离系数
    w0 = coefficients[:, :, 0]  # [n_samples, horizon]
    expert_weights = coefficients[:, :, 1:]  # [n_samples, horizon, n_experts]

    # 截距项统计
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

    # 专家权重统计
    for i in range(expert_weights.shape[2]):
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

    # 极端权重分析（超过[0,1]范围的权重）
    extreme_weights = {}
    for i in range(expert_weights.shape[2]):
        weight_i = expert_weights[:, :, i]

        # 找出超过[0,1]范围的权重
        negative_mask = weight_i < 0
        large_mask = weight_i > 1

        extreme_weights[f'expert_{i}'] = {
            'negative_count': np.sum(negative_mask),
            'negative_percentage': np.mean(negative_mask) * 100,
            'large_count': np.sum(large_mask),
            'large_percentage': np.mean(large_mask) * 100,
            'negative_indices': np.where(negative_mask),
            'large_indices': np.where(large_mask),
            'min_negative': np.min(weight_i) if np.sum(negative_mask) > 0 else 0,
            'max_large': np.max(weight_i) if np.sum(large_mask) > 0 else 1
        }

    analysis_results['extreme_weights'] = extreme_weights

    # 权重相关性分析
    if expert_weights.shape[2] >= 2:
        # 计算权重之间的相关性
        weights_reshaped = expert_weights.reshape(-1, expert_weights.shape[2])
        correlation_matrix = np.corrcoef(weights_reshaped.T)
        analysis_results['correlation_analysis']['weight_correlations'] = correlation_matrix

    return analysis_results


def main():
    """测试动态门控网络"""
    import numpy as np

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建测试数据
    batch_size = 32
    input_size = 96
    horizon = 24
    num_features = 7
    n_experts = 2

    # 模拟历史数据
    x_hist = torch.randn(batch_size, input_size, num_features)

    # 模拟专家预测
    expert1_pred = torch.randn(batch_size, horizon) * 0.5 + 2.0
    expert2_pred = torch.randn(batch_size, horizon) * 0.3 + 1.8
    expert_predictions = torch.stack([expert1_pred, expert2_pred], dim=2)

    # 创建门控网络
    gating_network = GatingNetwork(
        input_size=1,  # 只处理目标变量
        hidden_size=64,
        num_layers=2,
        horizon=horizon,
        n_experts=n_experts,
        dropout=0.1
    )

    # 创建动态融合模型
    # 使用虚拟的专家模型
    dummy_experts = [nn.Linear(num_features, horizon) for _ in range(n_experts)]

    fusion_model = DynamicFusionModel(
        gating_network=gating_network,
        input_size=input_size,
        horizon=horizon,
        n_experts=n_experts,
        num_features=num_features
    )

    # 测试前向传播
    with torch.no_grad():
        final_pred, coefficients = fusion_model(x_hist, expert_predictions)
        print(f"输入历史数据形状: {x_hist.shape}")
        print(f"专家预测形状: {expert_predictions.shape}")
        print(f"最终预测形状: {final_pred.shape}")
        print(f"系数形状: {coefficients.shape}")

        assert final_pred.shape == (batch_size, horizon), f"输出形状不匹配: {final_pred.shape}"
        assert coefficients.shape == (batch_size, horizon, n_experts + 1), f"系数形状不匹配: {coefficients.shape}"

        print("动态门控网络测试通过！")

        # 分析系数
        coeffs_numpy = coefficients.numpy()
        analysis = analyze_coefficients(coeffs_numpy)

        print(f"\n系数分析:")
        print(f"截距项范围: [{analysis['w0_stats']['min']:.4f}, {analysis['w0_stats']['max']:.4f}]")
        for i in range(n_experts):
            stats = analysis['expert_weights_stats'][f'expert_{i}']
            print(f"专家{i}权重范围: [{stats['min']:.4f}, {stats['max']:.4f}]")


if __name__ == "__main__":
    main()