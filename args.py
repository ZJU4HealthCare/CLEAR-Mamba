import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('MedMamba training and evaluation script', add_help=False)

    # 基础配置
    parser.add_argument('--save_name', default="Classifier", type=str,help='Name of the model to save')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    # 模型配置
    parser.add_argument('--model', default='medmamba_t', type=str, metavar='MODEL',
                        help='Name of model to train: medmamba_t, medmamba_s, medmamba_b')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--d_state', type=int, default=16,
                        help='State dimension for selective scan (default: 16)')

    # MedMamba特定配置
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size for image embedding')
    parser.add_argument('--in_chans', type=int, default=3,
                        help='Input image channels')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of classes for classification head')
    parser.add_argument('--depths',  type=int, nargs=4, metavar=('C1','C2','C3','C4'), default=[2, 2, 4, 2],
                        help='Depth of each stage for tiny version')
    parser.add_argument('--dims',  type=int, nargs=4, metavar=('C1','C2','C3','C4'), default=[96, 192, 384, 768],
                        help='Dimensions of each stage for tiny version')

    # Tiny版本配置
    parser.add_argument('--depths_t',  type=int, nargs=4, metavar=('C1','C2','C3','C4'), default=[2, 2, 4, 2],
                        help='Depth of each stage for tiny version')
    parser.add_argument('--dims_t',  type=int, nargs=4, metavar=('C1','C2','C3','C4'), default=[96, 192, 384, 768],
                        help='Dimensions of each stage for tiny version')
    
    # Small版本配置
    parser.add_argument('--depths_s',  type=int, nargs=4, metavar=('C1','C2','C3','C4'), default=[2, 2, 8, 2],
                        help='Depth of each stage for small version')
    parser.add_argument('--dims_s',  type=int, nargs=4, metavar=('C1','C2','C3','C4'), default=[96, 192, 384, 768],
                        help='Dimensions of each stage for small version')
    
    # Base版本配置
    parser.add_argument('--depths_b',  type=int, nargs=4, metavar=('C1','C2','C3','C4'), default=[2, 2, 12, 2],
                        help='Depth of each stage for base version')
    parser.add_argument('--dims_b',  type=int, nargs=4, metavar=('C1','C2','C3','C4'), default=[128, 256, 512, 1024],
                        help='Dimensions of each stage for base version')

    # HyperAD配置
    parser.add_argument('--hyper_ad', default=0, type=int,
                        help='Whether to use HyperAD')
    parser.add_argument('--reduction_ratio', type=int, default=4,
                        help='Reduction ratio for HyperAD feature extraction')
    parser.add_argument('--had_feat_dim', type=int, default=None,
                        help='HyperAD feature_dim; default None = use embed_dim')
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--p_drop', type=float, default=0.1)

    # EDL配置
    parser.add_argument('--EDL', default=0, type=int,
                        help='Whether to use EDL')
    parser.add_argument('--edl_mode', type=str, default='adaptive',
                        choices = ['fixed', 'linear', 'ema', 'adaptive'])
    parser.add_argument('--kl_coef', default=1e-2, type=float)
    parser.add_argument('--kl_scale', type=float, default=1.2)
    # linear/ema 所需
    parser.add_argument('--kl_start', type=float, default=0.0)
    parser.add_argument('--kl_end', type=float, default=1e-2)
    parser.add_argument('--kl_warmup_epochs', type=int, default=10)
    parser.add_argument('--kl_ema_beta', type=float, default=0.9)  # ema 模式的平滑系数




    # 优化器配置
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    
    # 其他训练配置
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default=None, type=str)

    return parser

def get_args():
    parser = argparse.ArgumentParser('MedMamba training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    return args 