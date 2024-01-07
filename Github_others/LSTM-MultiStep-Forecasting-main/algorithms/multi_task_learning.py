# -*- coding:utf-8 -*-
"""
@Time：2022/05/25 23:20
@Author：KI
@File：seq2seq.py
@Motto：Hungry And Humble
"""
import os
import sys


#多任务学习使用数据集mlt_data_1.csv和mlt_data_1.csv

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from data_process import nn_seq_mtl
from code.args import multi_task_args_parser
from model_train import mtl_train, load_data
from model_test import mtl_test

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/mtl.pkl'

if __name__ == '__main__':
    args = multi_task_args_parser()
    flag = 'mtl'
    # Dtr, Val, Dte, m, n = load_data(args, flag, args.batch_size)
    # mtl_train(args, Dtr, Val, LSTM_PATH)
    # mtl_test(args, Dte, LSTM_PATH, m, n)

    Dtr, Val, Dte, scaler = nn_seq_mtl(seq_len=(args.seq_len), B=args.batch_size, pred_step_size=(args.pred_step_size))
    mtl_train(args, Dtr, Val, LSTM_PATH)
    mtl_test(args, Dte, scaler, LSTM_PATH)