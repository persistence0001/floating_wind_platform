import torch

print("=== CUDA状态检测 ===")
print(f"PyTorch版本：{torch.__version__}")
print(f"CUDA是否可用：{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"可用GPU数量：{torch.cuda.device_count()}")
    print(f"GPU名称：{torch.cuda.get_device_name(0)}")
    print(f"当前使用GPU：{torch.cuda.current_device()}")
else:
    print("CUDA未启用或不可用")