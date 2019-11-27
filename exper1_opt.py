import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import regex as re
from time import time
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch.utils.data import Dataset, DataLoader
#from torchcrf import CRF
from torch.utils.data.sampler import SubsetRandomSampler
import random

from typing import *

from sklearn.metrics import confusion_matrix

from torch.nn import Parameter
from functools import wraps
from RULE import RULEs
from POSMap import POSMAP
from my_stuff_opt import *

num_trial = 50
cur_trial = 1
BS = 256 - 16
tags = {0:'I', 1:'B', 2:'O', 3:'<pad>'}
scheduler_n = 3
word_length = 84
early_stop_n = 5
max_size_char = 6
same_padding = True
use_BN = True
activation_func = True
input_channel = 1
nums_filter = [1]
output_size = 64
size_of_embedding = 300
pos_size = len(POSMAP)
num_char_encoding_size = 135
FCN = False
file_name = input('enter ur logname: ')
num_epoch = 30
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
Train = MyDataloader('clean_small_tr2.txt', 'label_small_tr2.txt', RULEs, \
                           word_length, '|', 'char_vec_dictionary.txt',max_size_char, \
                           'fasttext.th.vec', 300, device, 'pos_small_tr2.txt',POSMAP)
Test = MyDataloader('clean_small_te2.txt', 'label_small_te2.txt', RULEs, \
                           word_length, '|', 'char_vec_dictionary.txt',max_size_char, \
                           'fasttext.th.vec', 300, device, 'pos_small_te2.txt',POSMAP)

def objective(trial):
    global cur_trial
    cur_trial+=1
    cnt_idle = 0
    gru_fcn_dropout = trial.suggest_uniform('gru_fcn_dropout', 0, 0.6)
    gru_dropout = trial.suggest_uniform('gru_dropout', 0, 0.6)
    gru_out_dropout = trial.suggest_uniform('gru_out_dropout', 0, 0.6)
    LR = trial.suggest_uniform('LR', 5, 10)*10**(-4)
    grucrf_hidden_size = trial.suggest_categorical('grucrf_hidden_size', [64, 128])
    w_decay = trial.suggest_categorical('w_decay', [-4, 0])
    
    train_loader = DataLoader(Train, batch_size=BS, shuffle=True)
    test_loader = DataLoader(Test, batch_size=BS, shuffle=True)
    
    with open(file_name + '.txt', 'a', encoding='utf8') as f:
        f.write(f'gru_fcn_dropout: {gru_fcn_dropout}\n')
        f.write(f'gru_dropout: {gru_dropout}, gru_out_dropout: {gru_out_dropout}\n')
        f.write(f'LR: {LR}, grucrf_hidden_size: {grucrf_hidden_size}\n')
        f.write(f'w_decay: {w_decay}\n')

    NER = CNN_GRU_word_pos(BS, 300, word_length, grucrf_hidden_size, gru_dropout, True, \
        tags, gru_fcn_dropout, pos_size, gru_out_dropout)
    optimizer = optim.Adam(NER.parameters(), lr=LR, eps=1e-08, weight_decay=10**w_decay,amsgrad=True)
    my_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    print(device)
    NER.to(device)
    best_score = 0
    last_loss = 0
    for epoch in range(num_epoch):
        ttt = time()
        print(f'epoch {epoch}')
        all_loss = []
        for ind, batch_x in enumerate(train_loader):
            if ind%5 == 0:
                print(ind)
                print(f'cur_trial: {cur_trial}')
            t2 = time()
            NER = NER.train()
            print(time() - t2)
            NER.zero_grad()
            t1 = time()
            loss = NER(batch_x)
            loss = loss*(-1)
            print(f'time per batch: {time() - t1}')
            print(loss)
            all_loss.append(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(NER.parameters(), 5, norm_type=2)
            optimizer.step()
        total_loss = sum(all_loss)/(ind + 1)
        my_scheduler.step(total_loss)
        print(f'total loss of epoch: {total_loss.item()}')
        print('testing')
        per_mat = np.zeros((len(tags), 3))
        cnt_mat = np.zeros((len(tags), 3))
        for ind, batch_test in enumerate(test_loader):
            NER = NER.eval()
            output = NER.predict(batch_test)
            a, b = eval_score(tags, output, batch_test[2])
            per_mat += a
            cnt_mat += b
        per_mat = per_mat/(ind+1)
        per_mat = per_mat[:len(tags),:]
        cnt_mat = cnt_mat[:len(tags),:]
        print(cnt_mat)
        print(per_mat)
        score = sum(per_mat[:,2])/(len(tags)-1)

        with open(file_name + '.txt', 'a', encoding='utf8') as f:
            f.write(f'epoch: {epoch}, score: {score}\n')
        
        if last_loss - total_loss.item() >= 0.1:
            if best_score < score:
                best_score = score
                best_mat=per_mat
            if score > 0.33:
                cnt_idle = 0
            else:
                cnt_idle += 1
        else:
            cnt_idle += 1
        print(f'cnt_idle: {cnt_idle}')
        last_loss = total_loss.item()
        print(f'overall score: {score}')
        print('--------------------')
        if early_stop_n == cnt_idle:
            break
        print(f'total epoch time: {ttt-time()}')
    with open(file_name + '.txt', 'a', encoding='utf8') as f:
        f.write(f'cur_trial: {cur_trial}')
        f.write(f'cnt_mat\n')
        f.write(f'I => : {cnt_mat[0,0]}, : {cnt_mat[0,1]}, : {cnt_mat[0,2]}\n')
        f.write(f'B => : {cnt_mat[1,0]}, : {cnt_mat[1,1]}, : {cnt_mat[1,2]}\n')
        f.write(f'O => : {cnt_mat[2,0]}, : {cnt_mat[2,1]}, : {cnt_mat[2,2]}\n')
        f.write(f'best_score: {best_score}\n')
        f.write(f'I => recall: {best_mat[0,0]}, precision: {best_mat[0,1]}, , f1: {best_mat[0,2]}\n')
        f.write(f'B => recall: {best_mat[1,0]}, precision: {best_mat[1,1]}, , f1: {best_mat[1,2]}\n')
        f.write(f'O => recall: {best_mat[2,0]}, precision: {best_mat[2,1]}, , f1: {best_mat[2,2]}\n')
        f.write(f'----------------------------------\n')
    return best_score

study = optuna.study.load_study(storage='sqlite:///'+ file_name +'.db', study_name='test_optuna_' + file_name)
study.optimize(objective, n_trials=num_trial)