from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models


# 获取分类模型
def get_model(config):
    print('Initializing {} model (num_classes={})...'.format('lrcn', config.num_classes))
    from model.lstm import LSTMModel
    original_model = models.__dict__['resnet18'](pretrained=True)         # CNN特征提取器
    model = LSTMModel(
        original_model,
        num_classes=config.num_classes,
        hidden_size=512,
        fc_size=512
    )
    model = model.cuda()
    return model