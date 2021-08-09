from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from torch.utils.data import DataLoader


# 获取训练数据集
def get_training_set(config, spatial_transform, temporal_transform, target_transform):
    assert config.dataset in ['ucf101', 'hmdb51']
    if config.dataset == 'ucf101':
        training_data = UCF101(
            config.video_path,
            config.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif config.dataset == 'hmdb51':
        training_data = HMDB51(
            config.video_path,
            config.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    return training_data


# 获取验证数据集
def get_validation_set(config, spatial_transform, temporal_transform, target_transform):
    assert config.dataset in ['ucf101', 'hmdb51']
    # 设置为不进行验证状态
    if config.no_eval:
        return None
    if config.dataset == 'ucf101':
        validation_data = UCF101(
            config.video_path,
            config.annotation_path,
            'validation',
            config.num_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)
    elif config.dataset == 'hmdb51':
        validation_data = HMDB51(
            config.video_path,
            config.annotation_path,
            'validation',
            config.num_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)
    return validation_data


# 数据加载器
def get_data_loaders(config, train_transforms, validation_transforms=None):
    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))
    data_loaders = dict()
    # 定义数据传输管道
    dataset_train = get_training_set(
        config, train_transforms['spatial'],
        train_transforms['temporal'], train_transforms['target'])
    data_loaders['train'] = DataLoader(
        dataset_train, config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True)

    print('Found {} training examples'.format(len(dataset_train)))
    if not config.no_eval and validation_transforms:
        dataset_validation = get_validation_set(
            config, validation_transforms['spatial'],
            validation_transforms['temporal'], validation_transforms['target'])
        print('Found {} validation examples'.format(len(dataset_validation)))
        data_loaders['validation'] = DataLoader(
            dataset_validation, config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True)
    # 返回dataloader，包括训练集和验证集等
    return data_loaders