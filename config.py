import yaml
import torch

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 仅保留运行时判断的参数（如DEVICE），无需写在.yaml中
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    return config