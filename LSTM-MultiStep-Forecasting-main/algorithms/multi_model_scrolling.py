import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import mms_args_parser
from model_train import train, load_data
from model_test import mms_rolling_test

path = os.path.abspath(os.path.dirname(os.getcwd()))
dir_name = os.path.dirname(path)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
LSTM_PATH = [
        path+'/models/mms/0.pkl',
        path+'/models/mms/1.pkl',
        path+'/models/mms/2.pkl',
        path+'/models/mms/3.pkl',
        path+'/models/mms/4.pkl',
        path+'/models/mms/5.pkl',
        path+'/models/mms/6.pkl',
        path+'/models/mms/7.pkl',
        path+'/models/mms/8.pkl',
        path+'/models/mms/9.pkl',
        path+'/models/mms/10.pkl',
        path+'/models/mms/11.pkl',
    ]

if __name__ == '__main__':
    args = mms_args_parser()
    flag = 'mms'
    Dtrs, Vals, Dtes, m, n = load_data(args, flag, batch_size=args.batch_size)
    for Dtr, Val, path in zip(Dtrs, Vals, LSTM_PATH):
        train(args, Dtr, Val, path)
    Dtrs, Vals, Dtes, m, n = load_data(args, flag, batch_size=1)
    mms_rolling_test(args, Dtes, LSTM_PATH, m, n)