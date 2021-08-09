from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from model.i3d import InceptionI3D


# 获取分类模型
def get_model(config):
    assert config.model in ['i3d', 'cnn+lstm']
    print('Initializing {} model (num_classes={})...'.format(config.model, config.num_classes))
    if config.model == 'i3d':
        from model.i3d import get_fine_tuning_parameters    # 用于冻结一部分层，微调指定层的参数
        model = InceptionI3D(
            num_classes=config.num_classes,
            spatial_squeeze=True,
            final_endpoint='logits',
            in_channels=3,
            dropout_keep_prob=config.dropout_keep_prob
        )
    if 'cuda' in config.device:
        print('Moving model to CUDA device...')
        model = model.cuda()
        if config.checkpoint_path:
            # 恢复检查点文件
            print('Loading pretrained model {}'.format(config.checkpoint_path))
            assert os.path.isfile(config.checkpoint_path)
            checkpoint = torch.load(config.checkpoint_path)
            if config.model == 'i3d':
                pretrained_weights = checkpoint
            else:
                pretrained_weights = checkpoint['state_dict']
            model.load_state_dict(pretrained_weights)
            print('Replacing model logits with {} output classes.'.format(config.finetune_num_classes))
            if config.model == 'i3d':
                model.replace_logits(config.finetune_num_classes)
            # 开始确定需要微调的网络层
            assert config.model in ('i3d', 'resnet'), 'finetune params not implemented...'
            finetune_criterion = config.finetune_prefixes if config.model in ('i3d', 'resnet') else config.finetune_begin_index
            parameters_to_train = get_fine_tuning_parameters(model, finetune_criterion)
            return model, parameters_to_train
    else:
        raise ValueError('CPU training not supported.')
    return model, model.parameters()
