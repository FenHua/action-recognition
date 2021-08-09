import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):

        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=0,
            # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
            bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_size[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_size[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):

        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)

        if self._use_batch_norm:
            x = self.bn(x)

        if self._activation_fn is not None:
            x = self._activation_fn(x, inplace=True)

        return x


class InceptionModule(nn.Module):

    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=[1, 1, 1],
            padding=0,
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=[1, 1, 1],
            padding=0,
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=[3, 3, 3],
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=[1, 1, 1],
            padding=0,
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=[3, 3, 3],
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3],
            stride=(1, 1, 1),
            padding=0)

        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=[1, 1, 1], padding=0,
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


# Inception-v1 I3D 结构.
class InceptionI3D(nn.Module):
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'logits',
    )

    # 初始化函数，num_class表示类别数
    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='logits',
                 name='inception_i3d', in_channels=3, dropout_keep_prob=1.0):
        """
        初始化 I3D 模型.
        输入:
          num_classes: logit层输出类别的个数.
          spatial_squeeze: 是否压缩logits层的空间维度(用于返回，默认为真).
          final_endpoint: 最终拟输出的端,默认为logits
          name: 此模型的名称
        """
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3D, self).__init__()

        self._model_name = name
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self._dropout_rate = 1.0 - dropout_keep_prob

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.layers = {}
        end_point = 'Conv3d_1a_7x7'
        self.layers[end_point] = Unit3D(in_channels, 64, kernel_size=[7, 7, 7], stride=(2, 2, 2), padding=3,
                                        name=name + end_point)

        end_point = 'MaxPool3d_2a_3x3'
        self.layers[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)

        end_point = 'Conv3d_2b_1x1'
        self.layers[end_point] = Unit3D(64, 64, kernel_size=[1, 1, 1], padding=0, name=name + end_point)

        end_point = 'Conv3d_2c_3x3'
        self.layers[end_point] = Unit3D(64, 192, kernel_size=[3, 3, 3], padding=1, name=name + end_point)

        end_point = 'MaxPool3d_3a_3x3'
        self.layers[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)

        end_point = 'Mixed_3b'
        self.layers[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)

        end_point = 'Mixed_3c'
        self.layers[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)

        end_point = 'MaxPool3d_4a_3x3'
        self.layers[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)

        end_point = 'Mixed_4b'
        self.layers[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)

        end_point = 'Mixed_4c'
        self.layers[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)

        end_point = 'Mixed_4d'
        self.layers[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)

        end_point = 'Mixed_4e'
        self.layers[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)

        end_point = 'Mixed_4f'
        self.layers[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point)

        end_point = 'MaxPool3d_5a_2x2'
        self.layers[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)

        end_point = 'Mixed_5b'
        self.layers[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point)

        end_point = 'Mixed_5c'
        self.layers[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point)

        end_point = 'AvgPool_5'
        self.layers[end_point] = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))

        end_point = 'Dropout_5'
        self.layers[end_point] = nn.Dropout(self._dropout_rate, inplace=True)

        end_point = 'logits'
        self.layers[end_point] = Unit3D(
            in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
            kernel_size=[1, 1, 1], padding=0, activation_fn=None,
            use_batch_norm=False, use_bias=True, name=name + end_point)

        # Adds all the modules and performs weight initialization
        self._init_network()

    def _init_network(self):
        # Adding everything as module
        for layer_name, layer in self.layers.items():
            self.add_module(layer_name, layer)
        self._init_weights(self.modules())

    def _init_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        for layer_name, layer in self.layers.items():
            x = layer(x)
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        return x  # logits

    def trainable_params(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def replace_logits(self, num_classes, device='cuda:0'):
        self._num_classes = num_classes
        self.layers['logits'] = Unit3D(
            in_channels=384 + 384 + 128 + 128, output_channels=num_classes,
            kernel_size=[1, 1, 1], padding=0, activation_fn=None,
            use_batch_norm=False, use_bias=True, name=self._model_name + 'logits')

        self.logits = self.layers['logits']

        # Weight initialization for new logits layer
        self._init_weights(self.logits.modules())

        # Move to GPU
        if 'cuda' in device: self.logits.cuda()


def get_fine_tuning_parameters(model, ft_prefixes):
    assert isinstance(ft_prefixes, str)
    # 没有指定参数位置
    if ft_prefixes == '':
        return model.parameters()
    print('#' * 60)
    print('Setting finetuning layer prefixes: {}'.format(ft_prefixes))  # 微调指定层的参数
    ft_prefixes = ft_prefixes.split(',')
    parameters = []
    param_names = []
    for param_name, param in model.named_parameters():
        for prefix in ft_prefixes:
            if param_name.startswith(prefix):
                print('  Finetuning parameter: {}'.format(param_name))
                parameters.append({'params': param, 'name': param_name})
                param_names.append(param_name)
    for param_name, param in model.named_parameters():
        if param_name not in param_names:
            # This sames a lot of GPU memory...
            param.requires_grad = False
    return parameters
