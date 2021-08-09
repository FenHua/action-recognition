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
os.environ["CUDA_VISIBLE_DEVICES"] ='1'   # 运行环境

config = parse_opts()                        # 配置解析
config = prepare_output_dirs(config)         # 输出文件夹初始化
config = init_cropping_scales(config)        # 裁剪配置
config = set_lr_scheduling_policy(config)    # 学习率配置
# 均值和方差设置
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
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

model = model_factory.get_model(config)
model.cuda()
print("==> Loading existing model '{}' ".format('lrcn'))
model_info = torch.load(os.path.join('model/checkpoints/', '{}_save_best.pth'.format(config.dataset)))
model.load_state_dict(model_info['state_dict'])
model.eval()

print('==> Starting test.......')
steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
print(steps_in_epoch)
accuracies = np.zeros(steps_in_epoch, np.float32)
epoch_start_time = time.time()
for step, (clips, targets) in enumerate(data_loader):
    start_time = time.time()
    clips = clips.permute(2, 0, 1, 3, 4)  # test ????????????????
    clips = clips.cuda()
    targets = targets.cuda()
    logits = model.forward(clips)
    logits = torch.mean(logits, dim=1)  # 取均值
    _, preds = torch.max(logits, 1)
    correct = torch.sum(preds == targets.data)
    accuracy = correct.double() / config.batch_size
    examples_per_second = config.batch_size / float(time.time() - start_time)
    accuracies[step] = accuracy.item()
epoch_duration = float(time.time() - epoch_start_time)
epoch_avg_acc = np.mean(accuracies)
print('The average accuracy is: {}, the total runtime is: {}'.format(epoch_avg_acc,epoch_duration))