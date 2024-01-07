import os
import sys
from code.args import ms_args_parser
from util import train, test, load_data

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/multivariate_single_step.pkl'


if __name__ == '__main__':
    args = ms_args_parser()
    flag = 'ms'
    Dtr, Val, Dte, m, n = load_data(args, flag)  # 表示使用多变量单步长的数据
    train(args, Dtr, Val, LSTM_PATH)
    test(args, Dte, LSTM_PATH, m, n)
