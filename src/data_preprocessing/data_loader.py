"""
数据加载和预处理模块
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """浮式风机平台数据加载器"""

    def __init__(self, config_path: str):
        """
        初始化数据加载器

        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.data = None
        self.target_scaler = StandardScaler()
        self.covariate_scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """
        加载Excel数据文件

        Returns:
            加载的DataFrame
        """
        try:
            file_path = self.config['data']['file_path']
            logger.info(f"正在加载数据文件: {file_path}")

            # 读取Excel文件
            self.data = pd.read_excel(file_path)

            # 检查数据完整性
            self._check_data_integrity()

            logger.info(f"数据加载成功，形状: {self.data.shape}")
            logger.info(f"列名: {list(self.data.columns)}")

            return self.data

        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    def _check_data_integrity(self):
        """检查数据完整性"""
        required_cols = [self.config['data']['target_variable']] + self.config['data']['covariates']

        # 检查必需列是否存在
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

        # 检查缺失值
        missing_data = self.data[required_cols].isnull().sum()
        if missing_data.sum() > 0:
            logger.warning(f"发现缺失值:\n{missing_data[missing_data > 0]}")
            # 使用前向填充处理缺失值
            self.data[required_cols] = self.data[required_cols].fillna(method='ffill')
            logger.info("已使用前向填充处理缺失值")

        # 检查异常值
        self._detect_outliers(required_cols)

    def _detect_outliers(self, columns: List[str], threshold: float = 3.0):
        """
        检测并处理异常值

        Args:
            columns: 需要检查的列
            threshold: Z-score阈值
        """
        for col in columns:
            z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
            outlier_mask = z_scores > threshold
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                logger.warning(f"列 '{col}' 发现 {outlier_count} 个异常值 (|Z-score| > {threshold})")
                # 使用上下限截断处理异常值
                lower_bound = self.data[col].quantile(0.01)
                upper_bound = self.data[col].quantile(0.99)
                self.data[col] = self.data[col].clip(lower_bound, upper_bound)
                logger.info(f"已使用1%和99%分位数截断处理异常值")

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理：标准化和特征工程

        Returns:
            标准化后的目标变量和协变量
        """
        target_col = self.config['data']['target_variable']
        covariate_cols = self.config['data']['covariates']

        # 提取目标变量和协变量
        target_data = self.data[target_col].values.reshape(-1, 1)
        covariate_data = self.data[covariate_cols].values

        # 分别对目标变量和协变量进行标准化
        target_scaled = self.target_scaler.fit_transform(target_data)
        covariate_scaled = self.covariate_scaler.fit_transform(covariate_data)

        logger.info(
            f"目标变量标准化完成，均值: {self.target_scaler.mean_[0]:.4f}, 方差: {self.target_scaler.scale_[0]:.4f}")
        logger.info(f"协变量标准化完成，形状: {covariate_scaled.shape}")

        return target_scaled, covariate_scaled

    def create_sequences(self, target_data: np.ndarray, covariate_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列样本

        Args:
            target_data: 目标变量数据
            covariate_data: 协变量数据

        Returns:
            输入序列和目标序列
        """
        input_size = self.config['data']['input_size']
        horizon = self.config['data']['horizon']

        # 合并目标变量和协变量
        full_data = np.concatenate([target_data, covariate_data], axis=1)

        X, y = [], []

        # 创建滑动窗口样本
        for i in range(len(full_data) - input_size - horizon + 1):
            # 输入序列
            x_seq = full_data[i:(i + input_size)]
            # 目标序列（仅目标变量）
            y_seq = target_data[(i + input_size):(i + input_size + horizon)].flatten()

            X.append(x_seq)
            y.append(y_seq)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"创建序列样本完成: X形状 {X.shape}, y形状 {y.shape}")

        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        按时间顺序划分数据集

        Args:
            X: 输入数据
            y: 目标数据

        Returns:
            训练集、验证集、测试集
        """
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']

        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"数据划分完成:")
        logger.info(f"训练集: X{X_train.shape}, y{y_train.shape}")
        logger.info(f"验证集: X{X_val.shape}, y{y_val.shape}")
        logger.info(f"测试集: X{X_test.shape}, y{y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_processed_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                            output_dir: str = "results"):
        """
        保存处理后的数据

        Args:
            X_train, X_val, X_test: 输入数据
            y_train, y_val, y_test: 目标数据
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存数据
        np.save(f"{output_dir}/X_train.npy", X_train)
        np.save(f"{output_dir}/X_val.npy", X_val)
        np.save(f"{output_dir}/X_test.npy", X_test)
        np.save(f"{output_dir}/y_train.npy", y_train)
        np.save(f"{output_dir}/y_val.npy", y_val)
        np.save(f"{output_dir}/y_test.npy", y_test)

        # 保存标准化器
        import joblib
        joblib.dump(self.target_scaler, f"{output_dir}/target_scaler.pkl")
        joblib.dump(self.covariate_scaler, f"{output_dir}/covariate_scaler.pkl")

        logger.info(f"处理后的数据已保存到 {output_dir}")

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        逆变换目标变量

        Args:
            y_scaled: 标准化的目标变量

        Returns:
            原始尺度的目标变量
        """
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def main():
    """主函数用于测试数据加载器"""
    import logging
    logging.basicConfig(level=logging.INFO)

    # 初始化数据加载器
    data_loader = DataLoader("configs/config.yaml")

    # 加载数据
    data = data_loader.load_data()
    print(f"数据形状: {data.shape}")
    print(f"数据预览:\n{data.head()}")

    # 预处理数据
    target_scaled, covariate_scaled = data_loader.preprocess_data()
    print(f"目标变量形状: {target_scaled.shape}")
    print(f"协变量形状: {covariate_scaled.shape}")

    # 创建序列
    X, y = data_loader.create_sequences(target_scaled, covariate_scaled)
    print(f"输入序列形状: {X.shape}")
    print(f"目标序列形状: {y.shape}")

    # 划分数据
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)

    # 保存数据
    data_loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)

    print("数据预处理完成！")


if __name__ == "__main__":
    main()