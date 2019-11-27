import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import regex as re
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection
from torch.utils.data import Dataset, DataLoader, random_split
#from torchcrf import CRF
from torch.utils.data.sampler import SubsetRandomSampler
import random

from torch.nn.utils.rnn import PackedSequence
from typing import *

from sklearn.metrics import confusion_matrix
#from sklearn_crfsuite import metrics
from torchnlp.nn import WeightDropGRU
from RULE import RULEs
from POSMap import POSMAP
from my_stuff import *

BS = 256-64
tags = {0:'I', 1:'B', 2:'O', 3:'<pad>'}
scheduler_n = 2
word_length = 84
early_stop_n = 4
max_size_char = 10
num_search = 1000
num_epoch = 10
filename = input('Enter your filename')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data = MyDataloader('./clean84withpos.txt', './label84withpos.txt', RULEs, \
                    word_length, '|', 'char_vec_dictionary.txt',max_size_char, \
                    './fasttext.th.vec', 300, device, './pos_tag84withpos.txt',POSMAP)

tr, te = get_indices_random_val_test_split(len(data), 1, 0.2, True)
train_loader = DataLoader(data, batch_size=BS, sampler=tr)
test_loader = DataLoader(data, batch_size=BS, sampler=te)

for cur_ind in range(num_search):
    torch.cuda.empty_cache()
    grucrf_dropout = random.uniform(0.3,0.5)#0.5
    DO_FCN_GRUCRF = random.uniform(0.3,0.5)#0.5
    grucrf_hidden_size = random.choice([128])#5
    LR = 10**random.uniform(-4,-5)#0.001
    size_kernel = random.choice([3,5])
    with open(filename+'.txt', 'a', encoding='utf8') as f:
        f.write(f'cur_ind: {cur_ind}\n')
        f.write(f'grucrf_dropout: {grucrf_dropout}, DO_FCN_GRUCRF: {DO_FCN_GRUCRF}\n')
        f.write(f'grucrf_hidden_size: {grucrf_hidden_size}, LR: {LR}, size_kernel: {size_kernel}\n')

    NER = CNN_GRU_CRF(BS, max_size_char, [1], True, True, 1, \
                      [size_kernel], True, 135, 135,\
                      300, word_length, grucrf_hidden_size, grucrf_dropout, \
                      True, tags, DO_FCN_GRUCRF, len(POSMAP), False)

    optimizer = optim.Adam(NER.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=True)
    my_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    print(device)
    NER.to(device)
    best_score = 0
    best_mat = np.zeros((len(tags)-1,3))
    cnt_idle = 0
    for epoch in range(num_epoch):
        print(f'epoch {epoch}')
        all_loss = 0
        for ind, batch_x in enumerate(train_loader):
            if ind%5 == 0:
                print(ind)
            t2 = time()
            NER = NER.train()
            print(time() - t2)
            NER.zero_grad()
            t1 = time()
            loss = NER(batch_x)
            loss = loss*(-1)
            print(f'time per batch: {time() - t1}')
            print(loss)
            all_loss += loss
            loss.backward()
            nn.utils.clip_grad_value_(NER.parameters(), 10)
            optimizer.step()
        total_loss = all_loss/(ind + 1)
        my_scheduler.step(total_loss)
        print(f'total loss of epoch: {total_loss.item()}')
        print('testing')
        per_mat = np.zeros((len(tags), 3))
        for ind, batch_test in enumerate(test_loader):
            NER = NER.eval()
            output = NER.predict(batch_test)
            per_mat += eval_score(tags, output, batch_test[2])
        per_mat = per_mat/(ind+1)
        per_mat = per_mat[:len(tags),:]
        print(per_mat)
        score = sum(per_mat[:,2])/(len(tags)-1)
        if epoch%2 == 0:
            with open(filename+'.txt', 'a', encoding='utf8') as f:
                f.write(f'epoch: {epoch}, score: {score}\n')
        if best_score < score:
            best_mat=per_mat
            best_score = score
            cnt_idle = 0
        else:
            cnt_idle += 1
        print(f'overall score: {score}')
        print('--------------------')
        if early_stop_n == cnt_idle:
            break
    with open(filename+'.txt', 'a', encoding='utf8') as f:
        f.write(f'best_score: {best_score}\n')
        f.write(f'I => recall: {best_mat[0,0]}, precision: {best_mat[0,1]}, , f1: {best_mat[0,2]}\n')
        f.write(f'B => recall: {best_mat[1,0]}, precision: {best_mat[1,1]}, , f1: {best_mat[1,2]}\n')
        f.write(f'O => recall: {best_mat[2,0]}, precision: {best_mat[2,1]}, , f1: {best_mat[2,2]}\n')
        f.write(f'----------------------------------\n')