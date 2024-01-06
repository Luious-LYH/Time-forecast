import os
import sys
import torch
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import mmss_args_parser
from model_train import train, load_data
from model_test import m_test

# path = os.path.abspath(os.path.dirname(os.getcwd()))
# models_directory = path + '/models/mmss/'
# LSTM_PATH = [os.path.join(models_directory, f) for f in os.listdir(models_directory) if f.endswith('.pt')]

path = os.path.abspath(os.path.dirname(os.getcwd()))
dir_name = os.path.dirname(path)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

#创建列表，便于模型遍历
LSTM_PATH = [
        path+'/models/mmss/0.pkl',
        path+'/models/mmss/1.pkl',
        path+'/models/mmss/2.pkl',
        path+'/models/mmss/3.pkl',
        path+'/models/mmss/4.pkl',
        path+'/models/mmss/5.pkl',
        path+'/models/mmss/6.pkl',
        path+'/models/mmss/7.pkl',
        path+'/models/mmss/8.pkl',
        path+'/models/mmss/9.pkl',
        path+'/models/mmss/10.pkl',
        path+'/models/mmss/11.pkl',
    ]

# if __name__ == '__main__':
#     args = mmss_args_parser()
#     flag = 'mmss'
#     Dtr, Val, Dte, m, n = load_data(args, flag, args.batch_size)
#     train(args, Dtr, Val, LSTM_PATH)
#     m_test(args, Dte, LSTM_PATH, m, n)
if __name__ == '__main__':
    args = mmss_args_parser()
    flag = 'mmss'
    Dtrs, Vals, Dtes, m, n = load_data(args, flag, batch_size=args.batch_size)
    for Dtr, Val, path in zip(Dtrs, Vals, LSTM_PATH):
        train(args, Dtr, Val, path)
    Dtrs, Vals, Dtes, m, n = load_data(args, flag, batch_size=1)
    m_test(args, Dtes, LSTM_PATH, m, n)