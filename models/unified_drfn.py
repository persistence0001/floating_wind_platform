import torch
import torch.nn as nn
from models.patchtst import PatchTST
from models.nhits import NHITS
import config


def get_fft_amplitudes(x_ts):
    """计算时间序列前N个最大傅里叶振幅作为辅助任务目标"""
    x_fft = torch.fft.rfft(x_ts, dim=1)
    amplitudes = torch.abs(x_fft)
    top_k_amplitudes, _ = torch.topk(amplitudes, k=config.FFT_N_AMPLITUDES, dim=1)
    return top_k_amplitudes


def get_stats_features(x_ts):
    """计算时间序列的统计特征作为辅助任务目标"""
    mean = torch.mean(x_ts, dim=1, keepdim=True)
    std = torch.std(x_ts, dim=1, keepdim=True)
    maximum = torch.max(x_ts, dim=1, keepdim=True).values
    minimum = torch.min(x_ts, dim=1, keepdim=True).values
    return torch.cat([mean, std, maximum, minimum], dim=1)


class UnifiedDRFN(nn.Module):
    """统一的端到端DRFN模型"""

    def __init__(self, patchtst_params, nhits_params):
        super().__init__()
        # 初始化专家模型
        self.patchtst = PatchTST(**patchtst_params)
        self.nhits = NHITS(**nhits_params)

        # 辅助任务头
        self.patchtst_aux_head = nn.Linear(patchtst_params['horizon'], config.FFT_N_AMPLITUDES)
        self.nhits_aux_head = nn.Linear(nhits_params['horizon'], 4)  # 4个统计特征

        # 门控网络
        self.gating_lstm = nn.LSTM(
            input_size=1,
            hidden_size=config.models['gating_network']['hidden_size'],
            num_layers=config.models['gating_network']['num_layers'],
            batch_first=True
        )
        self.gating_dense = nn.Linear(
            config.models['gating_network']['hidden_size'],
            config.data['horizon'] * 3  # w0, w1, w2
        )

    def forward(self, x):
        # 专家模型主任务预测
        pred_p_main = self.patchtst(x)  # [batch, horizon]
        pred_n_main = self.nhits(x)  # [batch, horizon]

        # 辅助任务预测
        pred_p_aux = self.patchtst_aux_head(pred_p_main)
        pred_n_aux = self.nhits_aux_head(pred_n_main)

        # 门控网络生成动态系数
        target_idx = config.data['covariates'].index(config.data['target_variable'])
        x_y = x[:, :, target_idx].unsqueeze(-1)  # [batch, input_size, 1]
        _, (h_n, _) = self.gating_lstm(x_y)
        coeffs = self.gating_dense(h_n[-1]).view(-1, config.data['horizon'], 3)

        # 动态融合
        w0, w1, w2 = coeffs[:, :, 0:1], coeffs[:, :, 1:2], coeffs[:, :, 2:3]
        final_pred = w0 + w1 * pred_p_main.unsqueeze(-1) + w2 * pred_n_main.unsqueeze(-1)

        return {
            "final_pred": final_pred.squeeze(-1),
            "coeffs": coeffs,
            "pred_p_main": pred_p_main,
            "pred_n_main": pred_n_main,
            "pred_p_aux": pred_p_aux,
            "pred_n_aux": pred_n_aux
        }