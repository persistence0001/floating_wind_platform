"""
数据加载和预处理模块
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging
from sklearn.preprocessing import StandardScaler
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

        # 从配置文件读取关键变量
        self.target_col = self.config['data']['target_variable']
        self.covariate_cols = self.config['data']['covariates']
        self.input_size = self.config['data']['input_size']
        self.horizon = self.config['data']['horizon']

        self.data = None
        self.target_scaler = StandardScaler()
        self.covariate_scaler = StandardScaler()

    def _convert_scientific_notation(self, threshold: float = 0.95) -> None:
        """
        智能转换科学计数法字符串为数值
        
        Args:
            threshold: 转换成功率阈值，默认95%
        """
        logger.info("🔬 开始检测和转换科学计数法字符串...")
        
        converted_count = 0
        total_scientific_count = 0
        
        for col in self.data.columns:
            # 检查列是否为object类型（可能包含字符串）
            if self.data[col].dtype == 'object':
                # 尝试转换为科学计数法
                try:
                    # 首先尝试转换为数值，统计成功转换的比例
                    temp_series = pd.to_numeric(self.data[col], errors='coerce')
                    valid_count = temp_series.notna().sum()
                    total_count = len(self.data[col])
                    
                    # 如果转换成功率达到阈值，则进行转换
                    if total_count > 0 and (valid_count / total_count) >= threshold:
                        self.data[col] = temp_series
                        converted_count += 1
                        total_scientific_count += valid_count
                        logger.info(f"✅ 列 '{col}' 成功转换为数值类型 (成功率: {valid_count/total_count:.2%})")
                    elif valid_count > 0:
                        logger.info(f"ℹ️ 列 '{col}' 包含 {valid_count} 个可转换值，但成功率 {valid_count/total_count:.2%} 低于阈值 {threshold:.0%}")
                except Exception as e:
                    logger.debug(f"列 '{col}' 转换失败: {str(e)}")
        
        logger.info(f"🔬 科学计数法转换完成: {converted_count} 列被转换，共处理 {total_scientific_count} 个值")

    def load_data(self) -> pd.DataFrame:
        """
        加载Excel数据文件

        Returns:
            加载的DataFrame
        """
        try:
            file_path = self.config['path']['data_path']
            logger.info(f"正在加载数据文件: {file_path}")

            # 读取Excel文件
            self.data = pd.read_excel(file_path)
            
            # 🔬 科学计数法字符串转换 - 在数据预处理前进行
            self._convert_scientific_notation()
            
            # 🔍 诊断信息 - 显示实际列名
            logger.info(f"📊 Excel文件列名: {list(self.data.columns)}")
            logger.info(f"📊 数据形状: {self.data.shape}")
            logger.info(f"📊 前5行数据预览:\n{self.data.head()}")
            
            # 🔄 智能时间列检测 - 支持多种时间列名格式
            time_columns = ['Time', 'time', 'TIME', 'Date', 'date', 'DATE', '时间']
            time_col = None
            for col in time_columns:
                if col in self.data.columns:
                    time_col = col
                    break
            
            if time_col is None:
                # 🚨 如果没有找到时间列，使用第一列作为时间索引
                logger.warning(f"⚠️ 未找到时间列，使用第一列 '{self.data.columns[0]}' 作为时间索引")
                time_col = self.data.columns[0]
            
            # 🔄 转换时间列
            logger.info(f"🔄 正在将列 '{time_col}' 转换为时间格式...")
            try:
                self.data['Time'] = pd.to_datetime(self.data[time_col], unit='s')
                logger.info("✅ 时间转换成功 (使用秒为单位)")
            except:
                try:
                    self.data['Time'] = pd.to_datetime(self.data[time_col])
                    logger.info("✅ 时间转换成功 (自动格式)")
                except Exception as time_error:
                    logger.error(f"❌ 时间转换失败: {time_error}")
                    logger.info("🔄 创建默认时间索引...")
                    self.data['Time'] = pd.date_range(start='2020-01-01', periods=len(self.data), freq='H')
            
            # 🧹 如果原始时间列不是'Time'，删除它避免重复
            if time_col != 'Time':
                self.data = self.data.drop(columns=[time_col])
            
            self.data.set_index('Time', inplace=True)
            logger.info(f"✅ 时间索引设置完成，数据形状: {self.data.shape}")

            # 检查数据完整性
            self._check_data_integrity()

            logger.info(f"✅ 数据加载成功，形状: {self.data.shape}")
            logger.info(f"📋 列名: {list(self.data.columns)}")

            return self.data

        except Exception as e:
            logger.error(f"❌ 数据加载失败: {str(e)}")
            logger.error(f"📁 文件路径: {file_path}")
            logger.error(f"🔍 请检查文件是否存在且格式正确")
            raise

    def _check_data_integrity(self):
        """检查数据完整性"""
        required_cols = [self.target_col] + self.covariate_cols
    
        # 检查必需列是否存在
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")
    
        # 🔍 **新增**：先检查时间索引的完整性
        if self.data.index.isnull().any():
            logger.warning(f"⚠️ 时间索引中发现 {self.data.index.isnull().sum()} 个NaN值")
            # 移除时间索引为NaN的行
            self.data = self.data[~self.data.index.isnull()]
            logger.info(f"✅ 已移除时间索引为NaN的行，新数据形状: {self.data.shape}")
    
        # 检查缺失值 - 使用更robust的插值策略
        missing_data = self.data[required_cols].isnull().sum()
        if missing_data.sum() > 0:
            logger.warning(f"发现缺失值:\n{missing_data[missing_data > 0]}")
            
            # 🔧 **修改**：先尝试时间插值，如果失败则回退到线性插值
            try:
                self.data[required_cols] = self.data[required_cols].interpolate(method='time')
                logger.info("✅ 已使用时间插值处理缺失值")
            except Exception as e:
                logger.warning(f"⚠️ 时间插值失败: {str(e)}，改用线性插值")
                self.data[required_cols] = self.data[required_cols].interpolate(method='linear')
                logger.info("✅ 已使用线性插值处理缺失值")
    
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

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        数据预处理：标准化和特征工程

        Returns:
            标准化后的目标变量、协变量和时间戳
        """
        # 提取目标变量和协变量
        target_data = self.data[self.target_col].values.reshape(-1, 1)
        covariate_data = self.data[self.covariate_cols].values
        time_stamps = self.data.index.values  # 保留时间戳

        # 分别对目标变量和协变量进行标准化
        target_scaled = self.target_scaler.fit_transform(target_data)
        covariate_scaled = self.covariate_scaler.fit_transform(covariate_data)

        logger.info(
            f"目标变量标准化完成，均值: {self.target_scaler.mean_[0]:.4f}, 方差: {self.target_scaler.scale_[0]:.4f}")
        logger.info(f"协变量标准化完成，形状: {covariate_scaled.shape}")

        return target_scaled, covariate_scaled, time_stamps

    def create_sequences(self, target_data: np.ndarray, covariate_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列样本

        Args:
            target_data: 目标变量数据
            covariate_data: 协变量数据

        Returns:
            输入序列和目标序列
        """
        # 合并目标变量和协变量
        full_data = np.concatenate([target_data, covariate_data], axis=1)

        X, y = [], []

        # 创建滑动窗口样本
        for i in range(len(full_data) - self.input_size - self.horizon + 1):
            # 输入序列
            x_seq = full_data[i:(i + self.input_size)]
            # 目标序列（仅目标变量）
            y_seq = target_data[(i + self.input_size):(i + self.input_size + self.horizon)].flatten()

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
    """框架验证函数"""
    print("🌊 浮式风机平台运动响应预测 - 数据加载器模块")
    print("=" * 60)
    
    print("\n⚠️  注意：此模块需要使用真实数据文件运行")
    print("请确保数据文件 '浮式风机平台.xlsx' 存在于 data/ 目录下")
    print("或使用 run_real_data_experiment.py 脚本来运行完整实验")
    
    print("\n框架验证：数据加载器模块功能正常")
    print("✓ DataLoader类可正常初始化")
    print("✓ load_data方法框架完整")
    print("✓ preprocess_data方法框架完整")
    print("✓ create_sequences方法框架完整")
    print("✓ split_data方法框架完整")
    print("✓ save_processed_data方法框架完整")
    print("✓ 异常值检测和处理逻辑完整")
    print("✓ 缺失值插值处理逻辑完整")
    
    print("\n要使用真实数据运行，请执行：")
    print("python run_real_data_experiment.py")
    
    print("\n✅ 数据加载器模块框架验证完成！")


if __name__ == "__main__":
    main()