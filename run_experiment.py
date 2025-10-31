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

        # 快速测试模式 - 使用真实数据
        if args.quick:
            logger.info("运行快速验证模式 - 使用真实数据...")
            print("快速验证模式 - 使用真实数据验证系统功能")
            
            # 加载和预处理真实数据
            logger.info("步骤1: 加载真实数据...")
            experiment.load_and_preprocess_data()
            print("✓ 真实数据加载成功")
            
            # 使用默认参数快速训练模型（减少epoch）
            logger.info("步骤2: 快速训练模型...")
            original_epochs = experiment.config['training']['num_epochs']
            experiment.config['training']['num_epochs'] = 5  # 快速模式只训练5个epoch
            
            experiment.train_expert_models(use_optimized_params=False)
            experiment.get_expert_predictions()
            print("✓ 模型训练完成")
            
            # 快速运行融合策略（减少MPA迭代）
            logger.info("步骤3: 快速运行融合策略...")
            original_mpa_iterations = experiment.config['mpa']['max_iterations']
            experiment.config['mpa']['max_iterations'] = 20  # 快速模式只迭代20次
            experiment.config['mpa']['max_iterations'] = experiment.config['mpa'].get('quick_iterations', 20)
            #跑“完整”实验
           # max_iter = self.config['mpa']['max_iterations']
            
            experiment.implement_strategy_a()
            experiment.implement_strategy_b()
            
            # 恢复原始配置
            experiment.config['training']['num_epochs'] = original_epochs
            experiment.config['mpa']['max_iterations'] = original_mpa_iterations
            
            logger.info("快速验证模式完成！")
            print("✓ 融合策略运行完成")
            print("\n系统验证成功！可以运行完整实验以获得更精确的结果。")
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