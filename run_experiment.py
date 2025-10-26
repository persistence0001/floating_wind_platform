#!/usr/bin/env python3
"""
浮式风机平台运动响应预测实验启动脚本
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main import FloatingWindPlatformExperiment


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="浮式风机平台运动响应预测实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python run_experiment.py                    # 运行完整实验
    python run_experiment.py --no-optimize     # 跳过超参数优化
    python run_experiment.py --quick           # 快速测试模式
    python run_experiment.py --verbose         # 详细日志输出
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径 (默认: configs/config.yaml)'
    )

    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='跳过超参数优化，使用默认参数'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速测试模式，使用较少的数据和迭代'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细日志输出'
    )

    parser.add_argument(
        '--data-only',
        action='store_true',
        help='仅进行数据预处理'
    )

    parser.add_argument(
        '--models-only',
        action='store_true',
        help='仅训练专家模型'
    )

    parser.add_argument(
        '--fusion-only',
        action='store_true',
        help='仅运行融合策略'
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # 检查配置文件
        if not os.path.exists(args.config):
            logger.error(f"配置文件不存在: {args.config}")
            sys.exit(1)

        # 创建实验实例
        logger.info("初始化实验...")
        experiment = FloatingWindPlatformExperiment(args.config)

        # 快速测试模式
        if args.quick:
            logger.info("运行快速测试模式...")
            # 这里可以添加快速测试的逻辑
            print("快速测试模式 - 运行基本功能验证")
            from quick_test import main as quick_test_main
            quick_test_main()
            return

        # 仅数据预处理
        if args.data_only:
            logger.info("仅进行数据预处理...")
            experiment.load_and_preprocess_data()
            logger.info("数据预处理完成")
            return

        # 仅训练模型
        if args.models_only:
            logger.info("仅训练专家模型...")
            experiment.load_and_preprocess_data()
            if not args.no_optimize:
                experiment.optimize_expert_models()
            experiment.train_expert_models(use_optimized_params=not args.no_optimize)
            experiment.get_expert_predictions()
            logger.info("专家模型训练完成")
            return

        # 仅运行融合策略
        if args.fusion_only:
            logger.info("仅运行融合策略...")
            # 这里需要确保已有模型预测结果
            logger.error("融合策略需要先有专家模型预测结果，请先运行完整实验或模型训练")
            return

        # 运行完整实验
        logger.info("开始完整实验...")
        results = experiment.run_complete_experiment(optimize_hyperparameters=not args.no_optimize)

        # 打印结果摘要
        print("\n" + "=" * 60)
        print("实验完成！结果摘要：")
        print("=" * 60)

        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  MAE:  {metrics['MAE']:.6f}")
            print(f"  MAPE: {metrics['MAPE']:.6f}%")
            print(f"  R²:   {metrics['R2']:.6f}")
            print(f"  峰值RMSE: {metrics['peak_rmse']:.6f}")

        print(f"\n详细结果已保存到: {experiment.results_dir}")
        print("=" * 60)

        logger.info("实验成功完成！")

    except Exception as e:
        logger.error(f"实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()