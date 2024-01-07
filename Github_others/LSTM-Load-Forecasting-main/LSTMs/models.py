# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019, 00:11:34) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: E:\GitHub\LSTM-Load-Forecasting\models.py
# Compiled at: 2022-05-27 21:30:15
# Size of source mod 2**32: 3350 bytes
"""
@Time：2022/04/15 15:35
@Author：KI
@File：models.py
@Motto：Hungry And Humble
"""
from torch import nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM((self.input_size), (self.hidden_size), (self.num_layers), batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :] # 对于每一个批次，我们选择最后一个时间步的所有输出特征,张量维度：(batch_size, sequence_length, output_size)
        return pred


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM((self.input_size), (self.hidden_size), (self.num_layers), batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        output = output.contiguous().view(batch_size, seq_len, self.num_directions, self.hidden_size)
        output = torch.mean(output, dim=2)
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred