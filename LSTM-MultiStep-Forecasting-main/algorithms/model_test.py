# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019, 00:11:34) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: E:\GitHub\LSTM-MultiStep-Forecasting\model_test.py
# Compiled at: 2022-06-22 10:49:06
# Size of source mod 2**32: 14299 bytes
"""
@Time：2022/06/03 14:12
@Author：KI
@File：model_test.py
@Motto：Hungry And Humble
"""
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np, torch
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_process import device, get_mape, setup_seed, MyDataset
from model_train import load_data
from models import LSTM, BiLSTM, Seq2Seq, MTL_LSTM
setup_seed(20)

#test用于在测试集上评估训练好的神经网络模型

#直接多输出
def test(args, Dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    #加载保存的模型状态字典，并应用于当前模型
    model.load_state_dict(torch.load(path)['models'])
    #将模型设置为评估模式
    model.eval()
    print('predicting...')
    #预测
    for seq, target in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    plot(y, pred)

#多模型单步
def m_test(args, Dtes, PATHS, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    Dtes = [[x for x in iter(Dte)] for Dte in Dtes]
    models = []

    #遍历模型，要评估12个模型
    for path in PATHS:
        if args.bidirectional:
            model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
        else:
            model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
        model.load_state_dict(torch.load(path)['models'])
        model.eval()
        models.append(model)

    print('predicting...')
    for i in range(len(Dtes[0])):
        for j in range(len(Dtes)):
            model = models[j]
            seq, label = Dtes[j][i][0], Dtes[j][i][1]
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            seq = seq.to(device)
            with torch.no_grad():
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    plot(y, pred)

#seq2seq
def seq2seq_test(args, Dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for seq, target in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    plot(y, pred)

#将列表data中的数据每seq_len划分为一组，对应到本文中就是每12个batch的数据为一组。
def list_of_groups(data, sub_len):
    groups = zip(*(iter(data),) * sub_len)
    end_list = [list(i) for i in groups]
    count = len(data) % sub_len
    end_list.append(data[-count:]) if count != 0 else end_list
    return end_list

#单步滚动
def ss_rolling_test(args, Dte, path, m, n):
    """
    :param args:
    :param Dte:
    :param path:
    :param m:
    :param n:
    :return:
    """
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    Dte = [x for x in iter(Dte)]
    Dte = list_of_groups(Dte, args.pred_step_size)
    for sub_item in tqdm(Dte):
        sub_pred = []
        for seq_idx, (seq, label) in enumerate(sub_item, 0):
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            if seq_idx != 0:
                seq = seq.cpu().numpy().tolist()[0]
                if len(sub_pred) >= len(seq):
                    for t in range(len(seq)):
                        seq[t][0] = sub_pred[(len(sub_pred) - len(seq) + t)]

                else:
                    for t in range(len(sub_pred)):
                        seq[(len(seq) - len(sub_pred) + t)][0] = sub_pred[t]

            else:
                seq = seq.cpu().numpy().tolist()[0]
            seq = [seq]
            seq = torch.FloatTensor(seq)
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0)
            seq = [x for x in iter(seq)][0]
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                sub_pred.extend(y_pred)

        pred.extend(sub_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    plot(y, pred)

#多模型滚动
def mms_rolling_test(args, Dte, PATHS, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    models = []
    #遍历模型，要评估12个模型
    for path in PATHS:
        if args.bidirectional:
            model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
        else:
            model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
        model.load_state_dict(torch.load(path)['models'])
        model.eval()
        models.append(model)

    Dte = [x for x in iter(Dte)]
    Dte = list_of_groups(Dte, args.pred_step_size)
    for sub_item in tqdm(Dte):
        sub_pred = []
        for seq_idx, (seq, label) in enumerate(sub_item, 0):
            model = models[seq_idx]
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            if seq_idx != 0:
                seq = seq.cpu().numpy().tolist()[0]
                if len(sub_pred) >= len(seq):
                    for t in range(len(seq)):
                        seq[t][0] = sub_pred[(len(sub_pred) - len(seq) + t)]

                else:
                    for t in range(len(sub_pred)):
                        seq[(len(seq) - len(sub_pred) + t)][0] = sub_pred[t]

            else:
                seq = seq.cpu().numpy().tolist()[0]
            seq = [seq]
            seq = torch.FloatTensor(seq)
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0)
            seq = [x for x in iter(seq)][0]
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                sub_pred.extend(y_pred)

        pred.extend(sub_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    plot(y, pred)

#多任务学习
def mtl_test(args, Dte, scaler, path):
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = MTL_LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size), n_outputs=(args.n_outputs)).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.n_outputs)]
    preds = [[] for i in range(args.n_outputs)]
    for seq, targets in tqdm(Dte):
        targets = np.array(targets.data.tolist())
        for i in range(args.n_outputs):
            target = targets[:, i, :]
            target = list(chain.from_iterable(target))
            ys[i].extend(target)

        seq = seq.to(device)
        with torch.no_grad():
            _pred = model(seq)
            for i in range(_pred.shape[0]):
                pred = _pred[i]
                pred = list(chain.from_iterable(pred.data.tolist()))
                preds[i].extend(pred)

    ys, preds = np.array(ys).T, np.array(preds).T
    ys = scaler.inverse_transform(ys).T
    preds = scaler.inverse_transform(preds).T
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print(get_mape(y, pred))
        mtl_plot(y, pred, ind + 1)

    plt.show()

#绘图
def plot(y, pred):
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 500)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')
    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()

#多任务学习绘图
def mtl_plot(y, pred, ind):
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 500)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label=('true' + str(ind)))
    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label=('pred' + str(ind)))
    plt.grid(axis='y')
    plt.legend(loc='upper center', ncol=6)