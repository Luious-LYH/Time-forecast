# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019, 00:11:34) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: E:\GitHub\LSTM-Load-Forecasting\args.py
# Compiled at: 2022-06-22 11:46:10
# Size of source mod 2**32: 4203 bytes
"""
@Time：2022/04/15 15:30
@Author：KI
@File：args.py
@Motto：Hungry And Humble
"""
import argparse, torch

def mm_args_parser():
    parser = argparse.ArgumentParser()  # 创建了一个参数解析器对象 parser
    # add_argument() 方法为参数解析器添加各种命令行参数
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=7, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    args = parser.parse_args()  # parse_args() 方法用来解析参数
    return args  # 返回一个命名空间，包含传递给命令的所有参数


def ms_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='input dimension')
    parser.add_argument('--input_size', type=int, default=7, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    args = parser.parse_args()
    return args


def us_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='input dimension')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    args = parser.parse_args()
    return args