import yaml
import torch
from typing import Dict, Any

class ConfigValidator:
    """配置参数验证器 - 根本性解决类型安全问题"""
    
    # 参数类型定义
    PARAM_TYPES = {
        # 数据配置
        'data.input_size': int,
        'data.horizon': int,
        'data.num_features': int,
        'data.train_ratio': float,
        'data.val_ratio': float,
        'data.test_ratio': float,
        
        # 训练配置
        'training.batch_size': int,
        'training.num_epochs': int,
        'training.patience': int,
        'training.learning_rate': float,
        'training.weight_decay': float,
        'training.min_delta': float,
        
        # 模型配置
        'models.patchtst.dropout': float,
        'models.patchtst.head_dropout': float,
        'models.nhits.dropout': float,
        'models.gating_network.hidden_size': int,
        'models.gating_network.num_layers': int,
        'models.gating_network.dropout': float,
        
        # 优化配置
        'optimization.n_trials': int,
        'optimization.timeout': int,
        'optimization.tuning_epochs': int,
        'optimization.n_splits_cv': int,
        
        # MPA配置
        'mpa.population_size': int,
        'mpa.max_iterations': int,
        'mpa.quick_iterations': int,
        'mpa.fads_probability': float,
        'mpa.convergence_threshold': float,
        
        # 评估配置
        'evaluation.peak_percentage': float,
    }
    
    @staticmethod
    def validate_and_convert(config: Dict[str, Any]) -> Dict[str, Any]:
        """验证并转换配置参数类型"""
        validated_config = config.copy()
        
        for param_path, expected_type in ConfigValidator.PARAM_TYPES.items():
            section, key = param_path.split('.', 1)
            
            if section in validated_config and key in validated_config[section]:
                original_value = validated_config[section][key]
                
                try:
                    if expected_type == int:
                        converted_value = int(original_value)
                    elif expected_type == float:
                        converted_value = float(original_value)
                    else:
                        continue
                    
                    validated_config[section][key] = converted_value
                    print(f"✅ 配置参数 {param_path}: {original_value} -> {converted_value} ({expected_type.__name__})")
                    
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"配置参数 {param_path} 类型转换失败: "
                        f"期望{expected_type.__name__}，实际值{original_value} ({type(original_value).__name__})"
                    )
        
        return validated_config

def load_config(config_path: str) -> Dict[str, Any]:
    """加载并验证配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 验证并转换参数类型
    config = ConfigValidator.validate_and_convert(config)
    
    # 添加设备信息
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    return config