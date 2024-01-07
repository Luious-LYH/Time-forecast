# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019, 00:11:34) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: E:\GitHub\LSTM-MultiStep-Forecasting\model_train.py
# Compiled at: 2022-06-22 11:42:00
# Size of source mod 2**32: 7675 bytes
"""
@Time：2022/04/15 16:06
@Author：KI
@File：model_train.py
@Motto：Hungry And Humble
"""
import copy, os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from data_process import nn_seq_mmss, nn_seq_mo, nn_seq_sss, device, setup_seed
from models import LSTM, BiLSTM, Seq2Seq, MTL_LSTM
setup_seed(20)

#根据不同预测方法设置flag，再调用不同的数据处理方法
def load_data(args, flag, batch_size):
    if flag == 'mms' or flag == 'mmss':
        Dtr, Val, Dte, m, n = nn_seq_mmss(seq_len=(args.seq_len), B=batch_size, pred_step_size=(args.pred_step_size))
    else:
        if flag == 'mo' or flag == 'seq2seq':
            Dtr, Val, Dte, m, n = nn_seq_mo(seq_len=(args.seq_len), B=batch_size, num=(args.output_size))
        else:
            Dtr, Val, Dte, m, n = nn_seq_sss(seq_len=(args.seq_len), B=batch_size)
    return (
     Dtr, Val, Dte, m, n)

#计算损失
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

#计算多任务学习模型的损失
def get_mtl_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for seq, labels in Val:
        seq = seq.to(device)
        labels = labels.to(device)
        preds = model(seq)
        total_loss = 0
        for k in range(args.n_outputs):
            total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])

        total_loss /= preds.shape[0]
        val_loss.append(total_loss.item())

    return np.mean(val_loss)

#直接多输出，单步滚动，多模型单步，多模型滚动的训练函数
def train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size

    #初始化选择LSTM还是BiLSTM
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    #均方误差损失函数
    loss_function = nn.MSELoss().to(device)
    #选择使用Adam或SGD优化器
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam((model.parameters()), lr=(args.lr), weight_decay=(args.weight_decay))
    else:
        optimizer = torch.optim.SGD((model.parameters()), lr=(args.lr), momentum=0.9,
          weight_decay=(args.weight_decay))
    #初始化一个学习率调度器，用于在训练过程中调整学习率
    scheduler = StepLR(optimizer, step_size=(args.step_size), gamma=(args.gamma))
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    #训练循环
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
        if epoch > min_epochs:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)

#seq2seq的训练函数
def seq2seq_train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    batch_size = args.batch_size
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
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
        if epoch > min_epochs:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)

#多任务学习的训练函数
def mtl_train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = MTL_LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size), n_outputs=(args.n_outputs)).to(device)
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
        for seq, labels in Dtr:
            seq = seq.to(device)
            labels = labels.to(device)
            preds = model(seq)
            total_loss = 0
            for k in range(args.n_outputs):
                total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])

            total_loss /= preds.shape[0]
            train_loss.append(total_loss.item())
            total_loss.requires_grad_(True)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        val_loss = get_mtl_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)