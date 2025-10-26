"""
NHITS模型实现
基于分层插值和MLP的时间序列预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MLPBlock(nn.Module):
    """MLP块"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        # 残差连接
        self.residual = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) + self.residual(x)


class NHitsBlock(nn.Module):
    """NHits块"""

    def __init__(self,
                 input_size: int,
                 theta_size: int,
                 mlp_units: List[int],
                 pooling_size: int,
                 dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size
        self.theta_size = theta_size
        self.pooling_size = pooling_size

        # 最大池化层
        self.pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)

        # MLP层
        mlp_layers = []
        prev_size = input_size // pooling_size

        for hidden_size in mlp_units:
            mlp_layers.append(MLPBlock(prev_size, hidden_size, hidden_size, dropout))
            prev_size = hidden_size

        # 最后的输出层
        mlp_layers.append(nn.Linear(prev_size, theta_size))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, input_size] 输入序列

        Returns:
            theta: [batch_size, theta_size] 参数向量
        """
        # 添加通道维度用于池化
        x_pooled = self.pooling(x.unsqueeze(1)).squeeze(1)  # [batch_size, input_size // pooling_size]

        # MLP处理
        theta = self.mlp(x_pooled)  # [batch_size, theta_size]

        return theta


class BasisLayer(nn.Module):
    """基函数层"""

    def __init__(self, backcast_size: int, forecast_size: int, basis_type: str = 'trend'):
        super().__init__()

        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.basis_type = basis_type

        if basis_type == 'trend':
            # 多项式趋势基函数
            self.backcast_basis = self._create_polynomial_basis(backcast_size, degree=2)
            self.forecast_basis = self._create_polynomial_basis(forecast_size, degree=2)
        elif basis_type == 'seasonality':
            # 季节性基函数 (傅里叶基)
            self.backcast_basis = self._create_fourier_basis(backcast_size, n_harmonics=10)
            self.forecast_basis = self._create_fourier_basis(forecast_size, n_harmonics=10)
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

    def _create_polynomial_basis(self, size: int, degree: int) -> torch.Tensor:
        """创建多项式基函数"""
        basis = []
        for d in range(degree + 1):
            basis.append(torch.arange(size, dtype=torch.float32) ** d)
        return torch.stack(basis, dim=1)  # [size, degree+1]

    def _create_fourier_basis(self, size: int, n_harmonics: int) -> torch.Tensor:
        """创建傅里叶基函数"""
        basis = []
        t = torch.arange(size, dtype=torch.float32) / size

        for h in range(1, n_harmonics + 1):
            basis.append(torch.sin(2 * np.pi * h * t))
            basis.append(torch.cos(2 * np.pi * h * t))

        return torch.stack(basis, dim=1)  # [size, 2*n_harmonics]

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            theta: [batch_size, theta_size] 参数向量

        Returns:
            backcast: [batch_size, backcast_size]
            forecast: [batch_size, forecast_size]
        """
        device = theta.device

        if self.basis_type == 'trend':
            theta_backcast = theta[:, :3]  # 3个多项式系数
            theta_forecast = theta[:, 3:6] if theta.size(1) >= 6 else theta[:, :3]

            backcast_basis = self.backcast_basis.to(device)  # [backcast_size, 3]
            forecast_basis = self.forecast_basis.to(device)  # [forecast_size, 3]
        else:  # seasonality
            n_harmonics = 10
            theta_backcast = theta[:, :20]  # 20个傅里叶系数
            theta_forecast = theta[:, 20:40] if theta.size(1) >= 40 else theta[:, :20]

            backcast_basis = self.backcast_basis.to(device)  # [backcast_size, 20]
            forecast_basis = self.forecast_basis.to(device)  # [forecast_size, 20]

        # 计算backcast和forecast
        backcast = torch.matmul(theta_backcast, backcast_basis.t())  # [batch_size, backcast_size]
        forecast = torch.matmul(theta_forecast, forecast_basis.t())  # [batch_size, forecast_size]

        return backcast, forecast


class NHITSStack(nn.Module):
    """NHITS堆栈"""

    def __init__(self,
                 input_size: int,
                 horizon: int,
                 num_blocks: int,
                 num_layers: List[int],
                 mlp_units: List[int],
                 pooling_sizes: List[int],
                 dropout: float = 0.1,
                 stack_types: List[str] = ['trend', 'seasonality']):
        super().__init__()

        self.input_size = input_size
        self.horizon = horizon
        self.num_blocks = num_blocks
        self.stack_types = stack_types

        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            # 每个块有不同的池化大小
            pooling_size = pooling_sizes[i % len(pooling_sizes)]

            # 为每个块创建多个层
            for j, stack_type in enumerate(stack_types):
                # 确定theta_size
                if stack_type == 'trend':
                    theta_size = 6  # 3个backcast + 3个forecast系数
                else:  # seasonality
                    theta_size = 40  # 20个backcast + 20个forecast系数

                block = NHitsBlock(
                    input_size=input_size,
                    theta_size=theta_size,
                    mlp_units=mlp_units,
                    pooling_size=pooling_size,
                    dropout=dropout
                )

                basis_layer = BasisLayer(
                    backcast_size=input_size,
                    forecast_size=horizon,
                    basis_type=stack_type
                )

                self.blocks.append(nn.ModuleDict({
                    'nhits_block': block,
                    'basis_layer': basis_layer,
                    'stack_type': stack_type
                }))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [batch_size, input_size] 输入序列

        Returns:
            forecast: [batch_size, horizon] 预测结果
            backcast: [batch_size, input_size] 回测结果
        """
        batch_size = x.shape[0]

        # 初始化残差
        residuals = x.clone()

        # 累积预测
        total_forecast = torch.zeros(batch_size, self.horizon, device=x.device)

        for block_dict in self.blocks:
            # NHits块处理
            theta = block_dict['nhits_block'](residuals)

            # 基函数层处理
            backcast, forecast = block_dict['basis_layer'](theta)

            # 更新残差
            residuals = residuals - backcast

            # 累积预测
            total_forecast = total_forecast + forecast

        return total_forecast, residuals


class NHITS(nn.Module):
    """NHITS模型"""

    def __init__(self,
                 input_size: int,
                 horizon: int,
                 num_stacks: int = 3,
                 num_blocks: List[int] = [1, 1, 1],
                 num_layers: List[int] = [2, 2, 2],
                 mlp_units: List[int] = [512, 512],
                 pooling_sizes: List[int] = [8, 4, 1],
                 n_freq_downsample: List[int] = [4, 2, 1],
                 dropout: float = 0.1,
                 num_features: int = 7):
        super().__init__()

        self.input_size = input_size
        self.horizon = horizon
        self.num_stacks = num_stacks
        self.num_features = num_features

        # 输入投影层
        self.input_projection = nn.Linear(num_features, 1)

        # NHITS堆栈
        self.stacks = nn.ModuleList()

        for i in range(num_stacks):
            stack = NHITSStack(
                input_size=input_size,
                horizon=horizon,
                num_blocks=num_blocks[i],
                num_layers=num_layers[i:i + 1] * len(['trend', 'seasonality']),
                mlp_units=mlp_units,
                pooling_sizes=[pooling_sizes[i]],
                dropout=dropout,
                stack_types=['trend', 'seasonality']
            )
            self.stacks.append(stack)

        # 最终融合层
        self.fusion = nn.Linear(horizon * num_stacks, horizon)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, input_size, num_features]

        Returns:
            预测结果 [batch_size, horizon]
        """
        batch_size = x.shape[0]

        # 投影到单一维度
        x_projected = self.input_projection(x).squeeze(-1)  # [batch_size, input_size]

        # 所有堆栈的预测结果
        stack_predictions = []

        for stack in self.stacks:
            forecast, _ = stack(x_projected)
            stack_predictions.append(forecast)

        # 融合所有堆栈的预测
        if len(stack_predictions) > 1:
            stacked_predictions = torch.stack(stack_predictions, dim=2)  # [batch_size, horizon, num_stacks]
            flattened = stacked_predictions.reshape(batch_size, -1)  # [batch_size, horizon * num_stacks]
            final_prediction = self.fusion(flattened)  # [batch_size, horizon]
        else:
            final_prediction = stack_predictions[0]

        return final_prediction


class NHITSTrainer:
    """NHITS训练器"""

    def __init__(self, model: NHITS, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()

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

    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            loss.backward()
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

                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.scheduler.step(avg_loss)

        return avg_loss

    def predict(self, dataloader) -> np.ndarray:
        """预测"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'horizon': self.model.horizon,
                'num_stacks': self.model.num_stacks,
                'num_features': self.model.num_features
            }
        }, path)
        logger.info(f"模型已保存到: {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已从 {path} 加载")


def main():
    """测试NHITS模型"""
    import numpy as np

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建测试数据
    batch_size = 32
    input_size = 96
    horizon = 24
    num_features = 7

    X = torch.randn(batch_size, input_size, num_features)
    y = torch.randn(batch_size, horizon)

    # 创建模型
    model = NHITS(
        input_size=input_size,
        horizon=horizon,
        num_stacks=3,
        num_blocks=[1, 1, 1],
        num_layers=[2, 2, 2],
        mlp_units=[512, 512],
        pooling_sizes=[8, 4, 1],
        n_freq_downsample=[4, 2, 1],
        dropout=0.1,
        num_features=num_features
    )

    # 测试前向传播
    with torch.no_grad():
        output = model(X)
        print(f"输入形状: {X.shape}")
        print(f"输出形状: {output.shape}")
        print(f"预期输出形状: ({batch_size}, {horizon})")

        assert output.shape == (batch_size, horizon), f"输出形状不匹配: {output.shape}"
        print("NHITS模型测试通过！")


if __name__ == "__main__":
    main()