# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019, 00:11:34) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: E:\GitHub\LSTM-Load-Forecasting\util.py
# Compiled at: 2022-06-22 11:47:16
# Size of source mod 2**32: 4992 bytes
"""
@Time：2022/04/15 16:06
@Author：KI
@File：util.py
@Motto：Hungry And Humble
"""
import copy, os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from itertools import chain
import torch
from scipy.interpolate import make_interp_spline
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import LSTM, BiLSTM
from data_process import nn_seq_us, nn_seq_ms, nn_seq_mm, device, get_mape, setup_seed, MyDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
setup_seed(20)

# 根据flag选择处理数据的方法
def load_data(args, flag):
    if flag == 'us':
        Dtr, Val, Dte, m, n = nn_seq_us(seq_len=(args.seq_len), B=(args.batch_size))
    else:
        if flag == 'ms':
            Dtr, Val, Dte, m, n = nn_seq_ms(seq_len=(args.seq_len), B=(args.batch_size))
        else:
            Dtr, Val, Dte, m, n = nn_seq_mm(seq_len=(args.seq_len), B=(args.batch_size), num=(args.output_size))
    return (Dtr, Val, Dte, m, n)

# 计算验证集的损失值
def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for seq, label in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)

# 训练模型并保存最佳模型
def train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam((model.parameters()), lr=(args.lr), weight_decay=(args.weight_decay))
    else:
        optimizer = torch.optim.SGD((model.parameters()), lr=(args.lr), momentum=0.9,
          weight_decay=(args.weight_decay))
    scheduler = StepLR(optimizer, step_size=(args.step_size), gamma=(args.gamma))
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for seq, label in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_loss = get_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    try:
        #path = path.replace("\\", "/")
        torch.save(state, path)
    except Exception as e:
        print("Error: unable to save the model")
        print(e)

# 测试模型并进行预测
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
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')
    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'
    plt.savefig(r"E:\Github\Lstm-load-forecast\Github_others\LSTM-Load-Forecasting-main\results\LSTM-Load-Forecasting.png.png")
    plt.show()
    input("Press Enter to continue...")
    