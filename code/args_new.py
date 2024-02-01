# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019, 00:11:34) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: E:\GitHub\LSTM-MultiStep-Forecasting\args.py
# Compiled at: 2022-06-22 11:41:06
# Size of source mod 2**32: 8734 bytes
"""
@Time：2023/01/22 15:30
@Author：Luious
@File：args.py
@Motto：Hungry And Humble
"""
import argparse, torch

# 定义了一个参数解析器函数，用于解析参数，并在代码中使用这些参数进行模型训练
def us_args_parser():#单变量单步预测
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=75, help='input dimension')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension') #单变量特征，所以是1
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
    args = parser.parse_args(args=[])
    return args

def ms_args_parser(): #多变量单步预测(input_size=7)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=75, help='input dimension')
    parser.add_argument('--input_size', type=int, default=11, help='input dimension')#因为是多变量，所以是7
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
    args = parser.parse_args(args=[])
    return args

def mm_args_parser(): #多变量多步预测(ouput_size=4)
    parser = argparse.ArgumentParser()  
    # add_argument() 方法为参数解析器添加各种命令行参数
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=11, help='input dimension')
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
    args = parser.parse_args(args=[])
    return args  # 返回一个命名空间，包含传递给命令的所有参数

# single_step_scrolling 单步滚动预测
def sss_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=75, help='input dimension')
    parser.add_argument('--input_size', type=int, default=11, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    args = parser.parse_args(args=[])
    return args

# multiple_outputs 直接多输出
def mo_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=11, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.008, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')#控制是LSTM还是BiLSTM
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    args = parser.parse_args(args=[])
    return args

# multi_model_single_step 多模型单步预测
def mmss_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=12, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    args = parser.parse_args(args=[])
    return args

# multi_model_scrolling 多模型滚动预测
def mms_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=12, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    args = parser.parse_args(args=[])
    return args

# seq2seq
def seq2seq_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=12, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=12, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    args = parser.parse_args(args=[])
    return args

# multi_task_learning 多任务学习
def multi_task_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=75, help='input dimension')
    parser.add_argument('--input_size', type=int, default=3, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=12, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.008, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--n_outputs', type=int, default=3, help='n_outputs')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    args = parser.parse_args(args=[])
    return args

#CNN_LSTM多变量多步预测
def cl_mm_args_parser(): #多变量多步预测(ouput_size=4)
    parser = argparse.ArgumentParser()  
    # add_argument() 方法为参数解析器添加各种命令行参数
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--input_size', type=int, default=12, help='input dimension')
    parser.add_argument('--in_channels', type=int, default=7, help='input channels')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--out_channels', type=int, default=32, help='output channels')
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
    args = parser.parse_args(args=[])
    return args  # 返回一个命名空间，包含传递给命令的所有参数


#Transformer多变量多步预测
def trans_args_parser(): 
    parser = argparse.ArgumentParser()  
    # add_argument() 方法为参数解析器添加各种命令行参数
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--input_size', type=int, default=12, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--d_model', type=int, default=128, help='d_model')
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
    args = parser.parse_args(args=[])
    return args  # 返回一个命名空间，包含传递给命令的所有参数