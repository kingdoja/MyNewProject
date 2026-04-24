# train.py
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量以更好地管理GPU内存
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 清理GPU缓存
import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 直接运行训练工具
if __name__ == "__main__":
    # 导入并运行训练工具
    from tools.train import main
    import argparse
    import torch
    
    # 在训练前清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 创建参数解析器
    # 创建一个 ArgumentParser 对象，用于处理命令行参数
    parser = argparse.ArgumentParser()

    # 添加 -c/--config 参数，指定配置文件路径，这个参数是必需的(required=True)
    parser.add_argument('-c', '--config', type=str, required=True)

    # 添加 -r/--resume 参数，用于从检查点恢复训练
    # 这个参数的值应该是一个检查点文件的路径
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')

    # 添加 -t/--tuning 参数，用于从检查点进行微调
    # 与resume的区别可能是微调会冻结部分网络层或使用不同的学习率策略
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')

    # 添加 -d/--device 参数，指定训练设备，如'cuda:0'或'cpu'
    parser.add_argument('-d', '--device', type=str, help='device')

    # 添加 --seed 参数，设置随机种子，确保实验可重现
    parser.add_argument('--seed', type=int, help='exp reproducibility')

    # 添加 --use-amp 标志参数，如果指定则使用自动混合精度训练
    # action='store_true' 表示如果命令行中有这个参数，则将其值设为True，否则为False
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')

    # 添加 --output-dir 参数，指定输出目录，用于保存模型检查点、日志等
    parser.add_argument('--output-dir', type=str, help='output directory')

    # 添加 --summary-dir 参数，指定TensorBoard摘要文件的保存目录
    parser.add_argument('--summary-dir', type=str, help='tensorboard summary')

    # 添加 --test-only 标志参数，如果指定则只进行测试，不进行训练
    # default=False 表示默认不启用此选项
    parser.add_argument('--test-only', action='store_true', default=False)

    # 添加 -u/--update 参数，允许通过命令行更新配置文件中的设置
    # nargs='+' 表示可以接受一个或多个参数值
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # 添加 --print-method 参数，指定打印方法，默认使用Python内置的print函数
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')

    # 添加 --print-rank 参数，在多进程训练中指定哪个进程可以打印信息
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    # 添加 --local-rank 参数，用于分布式训练中的本地进程ID
    # 这个参数通常由分布式训练框架(如torch.distributed.launch)自动设置
    parser.add_argument('--local-rank', type=int, help='local rank id')
    
    # 解析参数
    args = parser.parse_args()
    
    # 运行训练
    main(args)