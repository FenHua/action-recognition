from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch.nn as nn
from utils.utils import *
from config import parse_opts
from datetime import datetime
import factory.data_factory as data_factory
import factory.model_factory as model_factory
from transforms.target_transforms import ClassLabel
from epoch_iterators import train_epoch, validation_epoch
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, CenterCrop

'''--------------------------------------配置和日志设置------------------------------------------'''

config = parse_opts()                        # 配置解析
config = prepare_output_dirs(config)         # 输出文件夹初始化
config = init_cropping_scales(config)        # 裁剪配置
config = set_lr_scheduling_policy(config)    # 学习率配置
# 均值和方差设置
mean = [0.39608,0.38182,0.35067]
std = [0.15199,0.14856,0.15698]
print_config(config)                         # 输出配置文件
write_config(config, os.path.join(config.save_dir, 'config.json'))  # 写入json文件

'''---------------------------------------初始化模型-------------------------------------------'''

device = torch.device(config.device)                  # 运行环境
model, parameters = model_factory.get_model(config)   # 获取模型以及需要更新的参数
# 设置转换函数
norm_method = Normalize(mean, std)
train_transforms = {
    'spatial':  Compose([MultiScaleRandomCrop(config.scales, config.spatial_size),
                         RandomHorizontalFlip(),
                         ToTensor(config.norm_value),
                         norm_method]),
    'temporal': TemporalRandomCrop(config.sample_duration),
    'target':   ClassLabel()
}   # 训练时的数据转换
validation_transforms = {
    'spatial':  Compose([CenterCrop(config.spatial_size),
                         ToTensor(config.norm_value),
                         norm_method]),
    'temporal': TemporalRandomCrop(config.sample_duration),
    'target':   ClassLabel()
}   # 测试时的数据转换

'''------------------------------------设置数据迭代器和参数优化器----------------------------------'''

data_loaders = data_factory.get_data_loaders(config, train_transforms, validation_transforms)
phases = ['train', 'validation'] if 'validation' in data_loaders else ['train']
# 优化器和loss函数设置
criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(config, parameters)
# 恢复优化器参数和设置开始训练点
if config.finetune_restore_optimizer:
    restore_optimizer_state(config, optimizer)
# 设置学习率
if config.lr_scheduler == 'plateau':
    assert 'validation' in phases
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', config.lr_scheduler_gamma, config.lr_plateau_patience)
else:
    milestones = [int(x) for x in config.lr_scheduler_milestones.split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, config.lr_scheduler_gamma)
# 记录最好的验证结果
val_acc_history = []
best_val_acc = 0.0
for epoch in range(config.start_epoch, config.num_epochs+1):
    # 依次进行训练和测试阶段
    for phase in phases:
        if phase == 'train':
            train_loss, train_acc, train_duration = train_epoch(
                config=config,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                data_loader=data_loaders['train'],
                epoch=epoch)
        elif phase == 'validation':
            val_loss, val_acc, val_duration = validation_epoch(
                config=config,
                model=model,
                criterion=criterion,
                device=device,
                data_loader=data_loaders['validation'],
                epoch=epoch)
            val_acc_history.append(val_acc)
    # 更新学习率
    if config.lr_scheduler == 'plateau':
        scheduler.step(val_loss)
    else:
        scheduler.step(epoch)

    print('#'*60)
    print('EPOCH {} SUMMARY'.format(epoch+1))
    print('Training Phase.')
    print('  Total Duration:              {} minutes'.format(int(np.ceil(train_duration / 60))))
    print('  Average Train Loss:          {:.3f}'.format(train_loss))
    print('  Average Train Accuracy:      {:.3f}'.format(train_acc))

    if 'validation' in phases:
        print('Validation Phase.')
        print('  Total Duration:              {} minutes'.format(int(np.ceil(val_duration / 60))))
        print('  Average Validation Loss:     {:.3f}'.format(val_loss))
        print('  Average Validation Accuracy: {:.3f}'.format(val_acc))

    if 'validation' in phases and val_acc > best_val_acc:
        checkpoint_path = os.path.join(config.checkpoint_dir, 'save_best.pth')
        save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
        print('Found new best validation accuracy: {:.3f}'.format(val_acc))
        print('Model checkpoint (best) written to:     {}'.format(checkpoint_path))
        best_val_acc = val_acc
    # 保存模型
    if epoch % config.checkpoint_frequency == 0:
        checkpoint_path = os.path.join(config.checkpoint_dir, 'save_{:03d}.pth'.format(epoch+1))
        save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
        print('Model checkpoint (periodic) written to: {}'.format(checkpoint_path))
        cleanup_checkpoint_dir(config)  # 删除旧的检查点文件
    # 提前终止条件
    if epoch > config.early_stopping_patience:
        last_val_acc = val_acc_history[-config.early_stopping_patience:]
        if all(acc < best_val_acc for acc in last_val_acc):
            # 所有的结果均小于这个最优结果
            print('Early stopping because validation accuracy has not '
                  'improved the last {} epochs.'.format(config.early_stopping_patience))
            break
print('Finished training.')
