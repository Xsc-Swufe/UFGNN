import torch
import torch.nn.functional as F
import time
from sklearn import metrics
import torch.utils.data as Data

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score, matthews_corrcoef


def cal_performance(pred, gold, smoothing=False):

    loss = cal_loss(pred, gold, smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)

    percision = metrics.precision_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    recall = metrics.recall_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='macro')
    f1_score = metrics.f1_score(gold.cuda().data.cpu().numpy(), pred.cuda().data.cpu().numpy(), average='weighted')

    n_correct = pred.eq(gold)
    n_correct = n_correct.sum().item()

    return loss, n_correct, percision, recall, f1_score

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = 2

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')
    return loss


import random
def train_epoch(model,training_data,train_y,optimizer, device, criterion,args):
    ''' Epoch operation in training phase'''
    model.train()
    seq_len = len(training_data)  # 计算训练数据集 x_train 的长度。
    train_seq = list(range(seq_len))[
                args.length:]
    random.shuffle(train_seq)

    n_count = 0
    total_loss = 0
    total_loss_count = 0
    train_y = train_y.to(device)

    batch_train = args.batch_size

    k=0
    for i in train_seq:
        X_train= training_data[i - args.length + 1: i + 1].to(device)
        if torch.isnan(X_train).any():
            print("X_train 中存在 NaN 值！")
        pred, p = model(X_train.float()) #输入Eod  torch.Size([10, 163, 8])
        G = p.mean(dim=(0, 1))
        mu_G = G.mean()
        sigma_G = G.std()
        omega= 1e-3
        rpmb_loss = omega * ((sigma_G / (mu_G + 1e-8)) ** 2)

        #print((pred[:, 0] > pred[:, 1]).sum())
        loss = criterion(pred, train_y[i])
        loss = loss + rpmb_loss
        k = k + 1


        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        #print_gradients(model,i)
        if total_loss_count % batch_train == batch_train-1 :
            #print((pred[:, 0] > pred[:, 1]).sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train-1  :
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

    return total_loss / total_loss_count






def print_gradients(model,i):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} - num: {i},Mean: {param.grad.mean()}, Std: {param.grad.std()}, Max: {param.grad.max()}")


def evaluate_epoch(model, x_eval, y_eval, optimizer, device, args):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[args.length:]
    preds = []
    trues = []
    y_eval = y_eval.to(device).to(torch.float32)




    for i in seq:
        X = x_eval[i - args.length + 1: i + 1].to(device).to(torch.float32)

        output, _ = model(X.float())

        output = output.detach().cpu().to(torch.float32)
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i].cpu().numpy())

    acc, auc = metrics(trues, preds)
    #print(trues[0], preds[0])
    return acc, auc




def metrics(trues, preds):
    trues = np.concatenate(trues, -1)
    preds = np.concatenate(preds, 0)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    auc = roc_auc_score(trues, preds[:, 1])  # Assumes preds[:, 1] contains the probability for the positive class
    mcc = matthews_corrcoef(trues, preds.argmax(-1))  # Calculates MCC based on true labels and predicted classes
    return acc, auc

from torch import nn
import math
def reset_parameters(named_parameters):

    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])


