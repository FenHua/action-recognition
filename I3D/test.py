from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from utils.utils import *
from config import parse_opts
from torch.utils.data import DataLoader
import factory.model_factory as model_factory
from transforms.target_transforms import ClassLabel
from factory.data_factory import get_validation_set
from transforms.temporal_transforms import TemporalCenterCrop
from transforms.spatial_transforms import Compose, Normalize,ToTensor, CenterCrop

config = parse_opts()                        # 配置解析
device = torch.device(config.device) 
# config = prepare_output_dirs(config)         # 输出文件夹初始化
config = init_cropping_scales(config)        # 裁剪配置
config = set_lr_scheduling_policy(config)    # 学习率配置
# 均值和方差设置
mean = [0.39608,0.38182,0.35067]
std = [0.15199,0.14856,0.15698]
# 设置转换函数
norm_method = Normalize(mean, std)
validation_transforms = {
    'spatial':  Compose([CenterCrop(config.spatial_size),
                         ToTensor(255),
                         norm_method]),
    'temporal': TemporalCenterCrop(config.sample_duration),
    'target':   ClassLabel()
}   # 测试时的数据转换
print('==> Loading validation dataset........')
val_data=get_validation_set(config, validation_transforms['spatial'], validation_transforms['temporal'], validation_transforms['target'])
data_loader = DataLoader(val_data, config.batch_size, shuffle=True,num_workers=config.num_workers, pin_memory=True)

model,_= model_factory.get_model(config)   # 获取模型以及需要更新的参数
print("==> Loading existing model '{}' ".format('i3d'))
model_info = torch.load(os.path.join('model/checkpoints/', '{}_save_best.pth'.format(config.dataset)))
model.load_state_dict(model_info['state_dict'])
model.eval()

print('==> Starting test.......')
steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
print(steps_in_epoch)
accuracies = np.zeros(steps_in_epoch, np.float32)
epoch_start_time = time.time()
for step, (clips, targets) in enumerate(data_loader):
    clips = clips.to(device)
    targets = targets.to(device)
    targets = torch.unsqueeze(targets, -1)
    logits = model.forward(clips)
    _, preds = torch.max(logits, 1)
    correct = torch.sum(preds == targets.data)
    accuracy = correct.double() / config.batch_size
    accuracies[step] = accuracy.item()
epoch_duration = float(time.time() - epoch_start_time)
epoch_avg_acc = np.mean(accuracies)
print('The average accuracy is: {}, the total runtime is: {}'.format(epoch_avg_acc,epoch_duration))