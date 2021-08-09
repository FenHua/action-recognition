import argparse


def parse_opts():

    parser = argparse.ArgumentParser()

    # 路经设置
    parser.add_argument('--video_path', type=str, required=True, help='Path to location of dataset videos')
    parser.add_argument('--annotation_path', type=str, required=False, help='Path to location of dataset annotation file')
    parser.add_argument('--save_dir', default='./output/', type=str, help='Where to save training outputs.')

    # 数据集
    parser.add_argument('--dataset', type=str, required=True, help='Dataset string (ucf101 | hmdb51)')
    parser.add_argument('--num_val_samples', type=int, default=1, help='Number of validation samples for each activity')
    parser.add_argument('--norm_value', default=255, type=int, help='Divide inputs by 255 or 1')
    parser.add_argument('--num_classes', default=101, type=int, help= 'Number of classes (ucf101: 101, hmdb51: 51)')
    parser.set_defaults(no_dataset_std=True)

    # 预处理设置
    parser.add_argument('--spatial_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=64, type=int, help='Temporal duration of inputs')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--num_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')

    # 模型设置
    parser.add_argument('--model', default='i3d', type=str, help='(i3d | cnn+lstm)')
    parser.add_argument('--dropout_keep_prob', default=1.0, type=float, help='Dropout keep probability')

    # 微调设置
    parser.add_argument('--checkpoint_path', default='', type=str, help='Checkpoint file (.pth) of previous training')
    parser.add_argument('--finetune_num_classes', default=36, type=int, help='Number of classes for fine-tuning. num_classes is set to the number when pretraining.')
    parser.add_argument('--finetune_prefixes', default='logits,Mixed_5', type=str, help='Prefixes of layers to finetune, comma seperated (only used by I3D).')
    parser.add_argument('--finetune_restore_optimizer', action='store_true', help='Whether to restore optimizer state')

    # 优化设置
    parser.add_argument('--optimizer', default='adam', type=str, help='Which optimizer to use (SGD | adam | rmsprop)')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='Initial learning rate (divided by 10 while training by lr-scheduler)')
    parser.add_argument('--lr_plateau_patience', default='10', type=int, help='Decay learning rate on plateauing of the validation loss (set -1 to disable)')
    parser.add_argument('--lr_scheduler_milestones', default='25,50', type=str, help='Learning rate scheduling, when to multiply learning rate by gamma')
    parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float, help='Learning rate decay factor')
    parser.add_argument('--early_stopping_patience', default=10, type=int, help='Early stopping patience (number of epochs)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch, only relevant for finetuning')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs to train for')

    # 日志
    parser.add_argument('--print_frequency', type=int, default=5, help='Print frequency in number of train steps')
    parser.add_argument('--checkpoint_frequency', type=int, default=1, help='Save checkpoint after this number of epochs')
    parser.add_argument('--checkpoints_num_keep', type=int, default=5, help='Number of checkpoints to keep')

    # Misc（其它设置）
    parser.add_argument('--device', default='cuda:0', help='Device string cpu | cuda:0')
    parser.add_argument('--history_steps', default=25, type=int, help='History of running average meters')
    parser.add_argument('--num_workers', default=6, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--no_eval', action='store_true', default=False, help='Disable evaluation')

    return parser.parse_args()
