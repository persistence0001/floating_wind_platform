"""
PatchTST模型实现
基于Transformer的时间序列预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class PatchEmbedding(nn.Module):
    """Patch嵌入层"""

    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.embedding = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, features]
        batch_size, seq_len, features = x.shape

        # 将序列分割成patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.patch_len // 2):  # 50%重叠
            patch = x[:, i:i + self.patch_len, :]  # [batch_size, patch_len, features]
            patches.append(patch)

        # 嵌入patches
        patch_embeddings = []
        for patch in patches:
            # 将每个patch的特征展平并嵌入
            patch_flat = patch.reshape(batch_size, -1)  # [batch_size, patch_len * features]
            # 如果维度不匹配，进行填充或截断
            if patch_flat.shape[1] != self.patch_len:
                if patch_flat.shape[1] < self.patch_len:
                    padding = torch.zeros(batch_size, self.patch_len - patch_flat.shape[1],
                                          device=patch_flat.device)
                    patch_flat = torch.cat([patch_flat, padding], dim=1)
                else:
                    patch_flat = patch_flat[:, :self.patch_len]

            embedded = self.embedding(patch_flat)  # [batch_size, d_model]
            patch_embeddings.append(embedded)

        # 堆叠所有patches
        patch_embeddings = torch.stack(patch_embeddings, dim=1)  # [batch_size, num_patches, d_model]

        return patch_embeddings


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, d_model: int, n_heads: int, num_layers: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x


class PredictionHead(nn.Module):
    """预测头"""

    def __init__(self, d_model: int, horizon: int, dropout: float = 0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        # 使用全局平均池化
        x = torch.mean(x, dim=1)  # [batch_size, d_model]
        return self.mlp(x)  # [batch_size, horizon]


class PatchTST(nn.Module):
    """PatchTST模型"""

    def __init__(self,
                 input_size: int,
                 horizon: int,
                 patch_len: int = 16,
                 stride: int = 8,
                 num_layers: int = 3,
                 n_heads: int = 8,
                 d_model: int = 128,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 head_dropout: float = 0.1,
                 num_features: int = 7):  # 1个目标变量 + 6个协变量
        super().__init__()

        self.input_size = input_size
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features

        # 输入投影层
        self.input_projection = nn.Linear(num_features, d_model)

        # Patch嵌入
        self.patch_embedding = PatchEmbedding(patch_len, d_model)

        # Transformer编码器
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        )

        # 预测头
        self.prediction_head = PredictionHead(d_model, horizon, head_dropout)

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

        # 输入投影
        x = self.input_projection(x)  # [batch_size, input_size, d_model]

        # Patch嵌入
        patches = self.patch_embedding(x)  # [batch_size, num_patches, d_model]

        # Transformer编码
        encoded = self.encoder(patches)  # [batch_size, num_patches, d_model]

        # 预测
        predictions = self.prediction_head(encoded)  # [batch_size, horizon]

        return predictions


class PatchTSTTrainer:
    """PatchTST训练器"""

    def __init__(self, model: PatchTST, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
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
                'patch_len': self.model.patch_len,
                'stride': self.model.stride,
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
    """测试PatchTST模型"""
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
    model = PatchTST(
        input_size=input_size,
        horizon=horizon,
        patch_len=16,
        stride=8,
        num_layers=3,
        n_heads=8,
        d_model=128,
        d_ff=256,
        dropout=0.1,
        head_dropout=0.1,
        num_features=num_features
    )

    # 测试前向传播
    with torch.no_grad():
        output = model(X)
        print(f"输入形状: {X.shape}")
        print(f"输出形状: {output.shape}")
        print(f"预期输出形状: ({batch_size}, {horizon})")

        assert output.shape == (batch_size, horizon), f"输出形状不匹配: {output.shape}"
        print("PatchTST模型测试通过！")


if __name__ == "__main__":
    main()