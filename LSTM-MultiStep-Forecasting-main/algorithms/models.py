# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019, 00:11:34) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: E:\GitHub\LSTM-MultiStep-Forecasting\models.py
# Compiled at: 2022-06-22 10:59:27
# Size of source mod 2**32: 6904 bytes
"""
@Time：2022/04/15 15:35
@Author：KI
@File：models.py
@Motto：Hungry And Humble
"""
from torch import nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义了四个模型，用于不同预测方法

#LSTM
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
        pred = pred[:, -1, :]
        return pred

#双向LSTM
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
        self.linear = nn.Linear(self.num_directions * self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred

#编码器，采用了LSTM将原始数据进行编码，然后将LSTM的最后的隐状态和单元状态返回。
class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM((self.input_size), (self.hidden_size), (self.num_layers), batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        return (
         h, c)

#解码器，解码器同样也由LSTM组成，不过解码器的初始的隐状态和单元状态是编码器的输出。此外，解码器每次输入都是上次的输出。
class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM((self.input_size), (self.hidden_size), (self.num_layers), batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, h, c):
        batch_size = input_seq.shape[0]
        input_seq = input_seq.view(batch_size, 1, self.input_size)
        output, (h, c) = self.lstm(input_seq, (h, c))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return (
         pred, h, c)

#seq2seq
class Seq2Seq(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.output_size = output_size
        self.Encoder = Encoder(input_size, hidden_size, num_layers, batch_size)
        self.Decoder = Decoder(input_size, hidden_size, num_layers, output_size, batch_size)

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        outputs = torch.zeros(batch_size, seq_len, self.output_size).to(device)
        for t in range(seq_len):
            _input = input_seq[:, t, :]
            output, h, c = self.Decoder(_input, h, c)
            outputs[:, t, :] = output

        return outputs[:, -1, :]

#多任务学习LSTM
class MTL_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, n_outputs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.lstm = nn.LSTM((self.input_size), (self.hidden_size), (self.num_layers), batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        preds = []
        pred1, pred2, pred3 = self.fc1(output), self.fc2(output), self.fc3(output)
        pred1, pred2, pred3 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :]
        pred = torch.stack([pred1, pred2, pred3], dim=0)
        return pred