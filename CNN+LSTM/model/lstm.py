import torch
import torch.nn as nn


# LRCN
class LSTMModel(nn.Module):

    def __init__(self,original_model,num_classes,hidden_size, fc_size,dropout=0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size   # 隐藏层数量
        self.num_classes = num_classes   # 类别数
        self.fc_size = fc_size           # 全连接层大小（连接CNN与LSTM）
        # 选择一个特征提取器
        self.features = nn.Sequential(*list(original_model.children())[:-1])   # 特征提取层
        for i, param in enumerate(self.features.parameters()):
            param.requires_grad = False
        self.fc_pre = nn.Sequential(nn.Linear(512, fc_size), nn.Dropout())
        self.rnn = nn.LSTM(input_size = fc_size,
                    hidden_size = hidden_size,
                    batch_first = True,dropout=dropout)                                       # LSTM
        self.fc = nn.Linear(hidden_size, num_classes)                         # logits

    #  CNN+LSTM
    def forward(self, inputs, hidden=None, steps=0):
        length = len(inputs)    # 帧数
        fs = torch.zeros(inputs[0].size(0), length, self.rnn.input_size).cuda()
        for i in range(length):
            f = self.features(inputs[i])
            f = f.view(f.size(0), -1)
            f = self.fc_pre(f)
            fs[:, i, :] = f
        outputs, hidden = self.rnn(fs, hidden)
        outputs = self.fc(outputs)
        return outputs