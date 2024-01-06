# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019, 00:11:34) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: E:\GitHub\LSTM-Load-Forecasting\data_process.py
# Compiled at: 2022-06-22 11:46:10
# Size of source mod 2**32: 5600 bytes
"""
@Time: 2022/03/01 20:11
@Author: KI
@File: data_process.py
@Motto: Hungry And Humble
"""
import os, random, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 用于设置随机种子，以确保在训练模型时的可重复性
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 读取数据，填充缺失值
def load_data():
    """
    :return:
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/data.csv'
    df = pd.read_csv(path, encoding='gbk')
    df.fillna((df.mean()), inplace=True)
    return df


class MyDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

# 多变量多步 处理数据
def nn_seq_mm(seq_len, B, num):
    print('data processing...')
    dataset = load_data()
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size, step_size):
        load = data[data.columns[1]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        # 根据给定的步长 step_size，从数据集中提取输入序列和对应的目标标签
        seq = []
        for i in range(0, len(data) - seq_len - num, step_size):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = [
                 load[j]]
                for c in range(2, 8):
                    x.append(data[j][c])

                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        return seq

    Dtr = process(train, B, step_size=1)
    Val = process(val, B, step_size=1)
    Dte = process(test, B, step_size=num)
    return (
     Dtr, Val, Dte, m, n)

# 多变量单步 处理数据
def nn_seq_ms(seq_len, B):
    print('data processing...')
    dataset = load_data()
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size):
        load = data[data.columns[1]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = [
                 load[j]]
                for c in range(2, 8):
                    x.append(data[j][c])

                train_seq.append(x)

            train_label.append(load[(i + seq_len)])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        return seq

    Dtr = process(train, B)
    Val = process(val, B)
    Dte = process(test, B)
    return (
     Dtr, Val, Dte, m, n)

# 单变量单步 处理数据
def nn_seq_us(seq_len, B):
    print('data processing...')
    dataset = load_data()
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size):
        load = data[data.columns[1]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = [
                 load[j]]
                train_seq.append(x)

            train_label.append(load[(i + seq_len)])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        return seq

    Dtr = process(train, B)
    Val = process(val, B)
    Dte = process(test, B)
    return (
     Dtr, Val, Dte, m, n)

# 计算 MAPE
def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))