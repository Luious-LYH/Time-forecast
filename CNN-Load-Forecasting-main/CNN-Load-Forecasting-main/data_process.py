# -*- coding:utf-8 -*-
"""
@Time: 2022/03/01 23:16
@Author: KI
@File: data_process.py
@Motto: Hungry And Humble
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
CNN_PATH = 'model/CNN.pkl'  # CNN模型保存路径

# 读取数据并使用均值填充缺失值
def load_data(file_name):
    df = pd.read_csv('data/' + file_name, encoding='gbk')
    df.fillna(df.mean(), inplace=True)

    return df

# 数据处理
def nn_seq(file_name, B):
    print('data processing...')
    dataset = load_data(file_name)
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data):
        load = data[data.columns[1]]  # 第二列是负荷
        data = data.values.tolist()
        load = (load - n) / (m - n)  # 归一化
        load = load.tolist()
        # 将过去24个时刻的特征和下一个时刻的目标值组合成一个样本
        seq = []
        for i in range(len(data) - 24):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(data[j][c])
                train_seq.append(x)
            train_label.append(load[i + 24])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)  # 封装数据

        seq = DataLoader(dataset=seq, batch_size=B, shuffle=True, num_workers=0, drop_last=True)  # 批量加载数据

        return seq

    Dtr = process(train)
    Val = process(val)
    Dte = process(test)

    return Dtr, Val, Dte, m, n


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

# 计算MAPE
def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))
