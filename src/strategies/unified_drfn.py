# C:\Temp\Pcharm\floating_wind_platform\src\strategies\unified_drfn.py (全新内容)
"""
统一的端到端动态回归融合网络 (UnifiedDRFN) 及其训练器
移植并增强自 PatchTST_Mullt 项目的核心思想
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# 导入项目内的专家模型
from models.patchtst import PatchTST
from models.nhits import NHITS


# --- 辅助任务的定义 ---
def get_fft_amplitudes(x_ts, n_amplitudes):
    """计算时间序列前N个最大傅里叶振幅"""
    x_fft = torch.fft.rfft(x_ts, dim=1)
    amplitudes = torch.abs(x_fft)
    top_k_amplitudes, _ = torch.topk(amplitudes, k=n_amplitudes, dim=1)
    return top_k_amplitudes


def get_stats_features(x_ts):
    """计算时间序列的统计特征"""
    mean = torch.mean(x_ts, dim=1, keepdim=True)
    std = torch.std(x_ts, dim=1, keepdim=True)
    maximum = torch.max(x_ts, dim=1, keepdim=True).values
    minimum = torch.min(x_ts, dim=1, keepdim=True).values
    return torch.cat([mean, std, maximum, minimum], dim=1)


# --- 统一的端到端模型 ---
class UnifiedDRFN(nn.Module):
    """
    统一的、端到端的、带有专家多样性辅助任务的DRFN模型
    """

    def __init__(self, patchtst_params: dict, nhits_params: dict, drfn_config: dict):
        super().__init__()
        # 实例化内部专家
        self.patchtst = PatchTST(**patchtst_params)
        self.nhits = NHITS(**nhits_params)

        self.config = drfn_config
        self.horizon = drfn_config['horizon']

        # 门控网络：使用全特征作为输入以获取更丰富的上下文
        self.gating_lstm = nn.LSTM(
            input_size=drfn_config['num_features'],
            hidden_size=drfn_config['lstm_hidden_size'],
            num_layers=drfn_config['lstm_num_layers'],
            batch_first=True
        )
        self.gating_dense = nn.Linear(
            drfn_config['lstm_hidden_size'],
            self.horizon * 3  # w0, w1, w2
        )

    def forward(self, x):
        # x shape: [batch, input_size, num_features]

        # 1. 专家模型并行预测（主任务+辅助任务）
        pred_p_main, pred_p_aux = self.patchtst(x, return_aux=True)
        pred_n_main, pred_n_aux = self.nhits(x, return_aux=True)

        # 2. 门控网络生成动态系数 (使用全特征)
        _, (h_n, _) = self.gating_lstm(x)
        coeffs = self.gating_dense(h_n[-1]).view(-1, self.horizon, 3)

        # 3. 动态回归融合
        w0, w1, w2 = coeffs[:, :, 0:1], coeffs[:, :, 1:2], coeffs[:, :, 2:3]
        final_pred = w0 + w1 * pred_p_main + w2 * pred_n_main

        # 统一返回所有输出
        outputs = {
            "final_pred": final_pred,
            "coeffs": coeffs,
            "pred_p_main": pred_p_main,
            "pred_n_main": pred_n_main,
            "pred_p_aux": pred_p_aux,
            "pred_n_aux": pred_n_aux
        }
        return outputs


# --- 复合损失函数 ---
class CompositeLoss(nn.Module):
    """复合损失函数，包含主任务、辅助任务和正交化损失"""

    def __init__(self, weights: dict):
        super().__init__()
        self.main_criterion = nn.MSELoss()
        self.aux_criterion = nn.MSELoss()
        self.w_main = weights.get('main', 1.0)
        self.w_aux_p = weights.get('aux_p', 0.1)
        self.w_aux_n = weights.get('aux_n', 0.1)
        self.w_ortho = weights.get('ortho', 0.1)
        self.fft_n_amplitudes = weights.get('fft_n_amplitudes', 10)

    def _calculate_ortho_loss(self, pred_p, pred_n):
        """计算两个专家预测之间的相关性作为惩罚项"""
        pred_p = pred_p.view(pred_p.size(0), -1)
        pred_n = pred_n.view(pred_n.size(0), -1)

        pred_p_centered = pred_p - torch.mean(pred_p, dim=1, keepdim=True)
        pred_n_centered = pred_n - torch.mean(pred_n, dim=1, keepdim=True)

        cosine_sim = F.cosine_similarity(pred_p_centered, pred_n_centered, dim=1)
        return torch.mean(cosine_sim ** 2)

    def forward(self, model_outputs, y_true, x_hist):
        # 主任务损失
        loss_main = self.main_criterion(model_outputs['final_pred'], y_true)

        # 辅助任务目标 (从输入x_hist中提取目标列)
        target_col_idx = 0  # 假设目标变量在第一列
        x_target = x_hist[:, :, target_col_idx]

        # PatchTST辅助任务损失 (FFT)
        target_p_aux = get_fft_amplitudes(x_target, self.fft_n_amplitudes)
        loss_p_aux = self.aux_criterion(model_outputs['pred_p_aux'], target_p_aux)

        # NHITS辅助任务损失 (Stats)
        target_n_aux = get_stats_features(x_target)
        loss_n_aux = self.aux_criterion(model_outputs['pred_n_aux'], target_n_aux)

        # 正交化损失
        loss_ortho = self._calculate_ortho_loss(
            model_outputs['pred_p_main'], model_outputs['pred_n_main']
        )

        # 加权求和
        total_loss = (self.w_main * loss_main +
                      self.w_aux_p * loss_p_aux +
                      self.w_aux_n * loss_n_aux +
                      self.w_ortho * loss_ortho)

        return {
            "total_loss": total_loss, "main_loss": loss_main,
            "patchtst_aux_loss": loss_p_aux, "nhits_aux_loss": loss_n_aux,
            "ortho_loss": loss_ortho
        }


# --- 端到端训练器 ---
class UnifiedDRFNTrainer:
    def __init__(self, model: UnifiedDRFN, device: str, loss_weights: dict):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = CompositeLoss(loss_weights).to(device)
        self.optimizer = None
        self.scheduler = None

    def setup_training(self, learning_rate: float, weight_decay: float):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

    def _run_epoch(self, data_loader, is_training=True):
        self.model.train(is_training)
        total_loss = 0

        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            with torch.set_grad_enabled(is_training):
                outputs = self.model(x_batch)
                loss_dict = self.loss_fn(outputs, y_batch, x_batch)
                loss = loss_dict['total_loss']

            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def train_model(self, train_loader, val_loader, num_epochs: int, patience: int, save_path: str):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self._run_epoch(train_loader, is_training=True)
            val_loss = self._run_epoch(val_loader, is_training=False)

            print(f"Epoch {epoch:03d}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(save_path, "unified_drfn_best.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs.")
                    break

        self.model.load_state_dict(torch.load(os.path.join(save_path, "unified_drfn_best.pth")))
        return best_val_loss

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds, all_coeffs, all_true, all_x = [], [], [], []

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)
                all_preds.append(outputs['final_pred'].cpu().numpy())
                all_coeffs.append(outputs['coeffs'].cpu().numpy())
                all_true.append(y_batch.numpy())
                all_x.append(x_batch.cpu().numpy())

        return np.concatenate(all_preds), np.concatenate(all_coeffs), np.concatenate(all_true), np.concatenate(all_x)

    def analyze_coefficients(coefficients: np.ndarray, covariates: np.ndarray, covariate_names: list,
                             time_steps: np.ndarray):
        """
        分析门控网络动态权重
        Args:
            coefficients: 门控系数 [样本数, 预测步长, 3]（w0, w1, w2）
            covariates: 协变量数据 [样本数, 输入长度, 特征数]
            covariate_names: 特征名称列表
            time_steps: 时间戳数组 [样本数]
        """
        # 计算各系数的时间序列趋势（按预测步长平均）
        w0_mean = np.mean(coefficients[:, :, 0], axis=1)  # 截距项均值
        w1_mean = np.mean(coefficients[:, :, 1], axis=1)  # PatchTST权重均值
        w2_mean = np.mean(coefficients[:, :, 2], axis=1)  # NHITS权重均值

        # 创建结果DataFrame
        result_df = pd.DataFrame({
            'time': time_steps,
            'w0': w0_mean,
            'w1_patchtst': w1_mean,
            'w2_nhits': w2_mean
        })

        # 1. 绘制权重时间序列图
        plt.figure(figsize=(12, 6))
        plt.plot(result_df['time'], result_df['w0'], label='截距项 w0', linestyle='--')
        plt.plot(result_df['time'], result_df['w1_patchtst'], label='PatchTST 权重 w1')
        plt.plot(result_df['time'], result_df['w2_nhits'], label='NHITS 权重 w2')
        plt.xlabel('时间')
        plt.ylabel('门控系数值')
        plt.title('动态权重时间序列趋势')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('dynamic_weights_trend.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 计算权重与协变量的相关性
        covariate_mean = np.mean(covariates[:, :, :], axis=1)  # 协变量时间序列均值
        covariate_df = pd.DataFrame(covariate_mean, columns=covariate_names)
        corr_df = pd.concat([result_df[['w0', 'w1_patchtst', 'w2_nhits']], covariate_df], axis=1).corr()

        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='相关系数')
        plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr_df.index)), corr_df.index)
        plt.title('门控权重与协变量相关性热力图')
        for i in range(len(corr_df.index)):
            for j in range(len(corr_df.columns)):
                plt.text(j, i, f'{corr_df.iloc[i, j]:.2f}', ha='center', va='center',
                         color='white' if abs(corr_df.iloc[i, j]) > 0.5 else 'black')
        plt.savefig('weights_covariate_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 保存结果到CSV
        result_df.to_csv('dynamic_weights_analysis.csv', index=False)
        corr_df.to_csv('weights_correlation.csv')
        print("动态权重分析完成，已保存图表和CSV文件")

    # 微调：UnifiedDRFNTrainer的evaluate方法（确保返回格式适配分析函数，约第220行）
    def evaluate(self, data_loader):
        self.model.eval()
        all_preds, all_coeffs, all_true, all_x = [], [], [], []

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)
                all_preds.append(outputs['final_pred'].cpu().numpy())
                all_coeffs.append(outputs['coeffs'].cpu().numpy())  # 保持系数维度 [batch, horizon, 3]
                all_true.append(y_batch.numpy())
                all_x.append(x_batch.cpu().numpy())

        # 确保返回数组维度正确
        return (
            np.concatenate(all_preds, axis=0),
            np.concatenate(all_coeffs, axis=0),
            np.concatenate(all_true, axis=0),
            np.concatenate(all_x, axis=0)
        )

    def test_expert_aux_outputs(patchtst_model, nhits_model, test_loader, device):
        """
        阶段一测试：验证专家模型辅助任务输出格式和维度
        """
        print("=== 阶段一：测试专家模型辅助任务输出 ===")
        patchtst_model.eval()
        nhits_model.eval()
        with torch.no_grad():
            for x_batch, _ in test_loader:
                x_batch = x_batch.to(device)
                # 验证PatchTST输出
                pred_p_main, pred_p_aux = patchtst_model(x_batch, return_aux=True)
                assert pred_p_main.ndim == 2, f"PatchTST主预测维度错误，应为[batch, horizon]，实际{pred_p_main.ndim}"
                assert pred_p_aux.ndim == 2, f"PatchTST辅助预测维度错误，应为[batch, n_amplitudes]，实际{pred_p_aux.ndim}"
                # 验证NHITS输出
                pred_n_main, pred_n_aux = nhits_model(x_batch, return_aux=True)
                assert pred_n_main.ndim == 2, f"NHITS主预测维度错误，应为[batch, horizon]，实际{pred_n_main.ndim}"
                assert pred_n_aux.ndim == 2, f"NHITS辅助预测维度错误，应为[batch, 4]（统计特征），实际{pred_n_aux.ndim}"
                assert pred_n_aux.shape[1] == 4, f"NHITS辅助预测特征数错误，应为4，实际{pred_n_aux.shape[1]}"
                print(f"✅ 专家模型输出维度验证通过")
                print(f"  - PatchTST：主预测{pred_p_main.shape}，辅助预测{pred_p_aux.shape}")
                print(f"  - NHITS：主预测{pred_n_main.shape}，辅助预测{pred_n_aux.shape}")
                break  # 仅验证1个batch即可
        print("=== 阶段一测试完成 ===")

        def test_composite_loss(drfn_model, test_loader, device, loss_weights):
            """
            阶段二测试：验证复合损失函数各组件计算正常
            """
            print("=== 阶段二：测试复合损失函数 ===")
            drfn_model.eval()
            loss_fn = CompositeLoss(loss_weights).to(device)
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = drfn_model(x_batch)
                    loss_dict = loss_fn(outputs, y_batch, x_batch)
                    # 验证损失值合理性（非NaN、非无穷大）
                    assert not torch.isnan(loss_dict['total_loss']).any(), "总损失包含NaN"
                    assert not torch.isinf(loss_dict['total_loss']).any(), "总损失包含无穷大"
                    assert loss_dict['ortho_loss'] >= 0, "正交化损失应为非负值"
                    # 输出各损失组件值
                    print(f"✅ 复合损失计算正常")
                    print(f"  - 主损失：{loss_dict['main_loss'].item():.4f}")
                    print(f"  - PatchTST辅助损失：{loss_dict['patchtst_aux_loss'].item():.4f}")
                    print(f"  - NHITS辅助损失：{loss_dict['nhits_aux_loss'].item():.4f}")
                    print(f"  - 正交化损失：{loss_dict['ortho_loss'].item():.4f}")
                    print(f"  - 总损失：{loss_dict['total_loss'].item():.4f}")
                    break
            print("=== 阶段二测试完成 ===")

        def test_dynamic_weight_analysis(drfn_trainer, test_loader, covariate_names, time_steps):
            """
            阶段三测试：验证动态权重分析功能
            """
            print("=== 阶段三：测试动态权重分析 ===")
            # 先执行评估获取系数
            all_preds, all_coeffs, all_true, all_x = drfn_trainer.evaluate(test_loader)
            # 验证系数维度
            assert all_coeffs.ndim == 3, f"系数维度错误，应为[样本数, horizon, 3]，实际{all_coeffs.ndim}"
            assert all_coeffs.shape[2] == 3, f"系数数量错误，应为3（w0,w1,w2），实际{all_coeffs.shape[2]}"
            # 调用分析函数
            analyze_coefficients(all_coeffs, all_x, covariate_names, time_steps)
            print("✅ 动态权重分析执行完成，已生成图表和CSV文件")
            print("=== 阶段三测试完成 ===")

