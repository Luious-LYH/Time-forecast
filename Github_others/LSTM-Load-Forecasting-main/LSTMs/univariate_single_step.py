# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/18 14:27
@Author ：KI 
@File ：univariate_single_step.py
@Motto：Hungry And Humble

"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from util import train, test, load_data
from args import us_args_parser
from data_process import setup_seed

setup_seed(20)
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LSTM_PATH = os.path.join(path, 'models', 'univariate_single_step.pkl')
LSTM_PATH = LSTM_PATH.replace("\\", "/") #torch.save()不能保存windows路径，需要替换
# print(LSTM_PATH)


if __name__ == '__main__':
    args = us_args_parser()
    flag = 'us'
    Dtr, Val, Dte, m, n = load_data(args, flag)  # 表示使用单变量单步长的数据
    train(args, Dtr, Val, LSTM_PATH)
    test(args, Dte, LSTM_PATH, m, n)
