import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import sss_args_parser
from model_train import train, load_data
from model_test import ss_rolling_test

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/single_step_scrolling.pkl'

if __name__ == '__main__':
    args = sss_args_parser()
    flag = 'sss'
    Dtr, Val, Dte, m, n = load_data(args, flag, args.batch_size)
    train(args, Dtr, Val, LSTM_PATH)
    ss_rolling_test(args, Dte, LSTM_PATH, m, n)