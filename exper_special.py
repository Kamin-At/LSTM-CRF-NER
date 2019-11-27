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
from torch.utils.data import Dataset, DataLoader
#from torchcrf import CRF
from torch.utils.data.sampler import SubsetRandomSampler
import random

from torch.nn.utils.rnn import PackedSequence
from typing import *
from torchnlp.nn import WeightDropGRU

from sklearn.metrics import confusion_matrix
#from sklearn_crfsuite import metrics
from RULE import RULEs
from POSMap import POSMAP
from my_stuff4_2gru import *

file_name = input('Enter ur log name: ')
num_epoch = 20

for IND in range(2):
    
    BS = 256+64
    tags = {0:'I', 1:'B', 2:'O', 3:'<pad>'}
    scheduler_n = 2
    word_length = 84
    early_stop_n = 3
    max_size_char = [6]#[5, 10, 20]
    nums_filter = [1]
    use_BN = True
    activation_func = True
    input_channel = 1
    kernel_sizes = [5]
    same_padding = True
    num_char_encoding_size = 135
    output_size = 64
    size_of_embedding = 300
    pos_size = len(POSMAP)
    FCN = False
    grucrf_dropout = [0.4]#[0, 0.15, 0.30, 0.45, 0.60]
    total_search = len(max_size_char)*len(grucrf_dropout)*2
    for size_char in max_size_char:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(device)

        data_tr = MyDataloader('clean84withpos_ne_tr'+ str(IND) +'.txt', 'label84withpos_ne_tr'+ str(IND) +'.txt',\
                               RULEs, word_length, '|', 'char_vec_dictionary.txt', size_char, \
                               'fasttext.th.vec', 300, device, 'pos_tag84withpos_ne_tr'+ str(IND) +'.txt',POSMAP)
        data_te = MyDataloader('clean84withpos_ne_te'+ str(IND) +'.txt', 'label84withpos_ne_te'+ str(IND) +'.txt', \
                               RULEs, word_length, '|', 'char_vec_dictionary.txt', size_char, \
                               'fasttext.th.vec', 300, device, 'pos_tag84withpos_ne_te'+ str(IND) +'.txt',POSMAP)

        train_loader = DataLoader(data_tr, batch_size=BS, shuffle= True)
        test_loader = DataLoader(data_te, batch_size=BS, shuffle= True)
        
        torch.cuda.empty_cache()
        for i in grucrf_dropout:
            
            grucrf_hidden_size = 128
            LR = 8*10**(-5)
            with open(file_name + '.txt', 'a', encoding='utf8') as f:
                f.write(f'lstmcrf_dropout = DO_FCN_LSTMCRF: {i}, size_char: {size_char}\n')
                f.write(f'lstmcrf_hidden_size: {grucrf_hidden_size}, LR: {LR}\n')
            print(f'lstmcrf_dropout = DO_FCN_LSTMCRF: {i}')
            print(f'lstmcrf_hidden_size: {grucrf_hidden_size}, LR: {LR}')

            NER = CNN_GRU_char_pos(BS, size_char, nums_filter, use_BN, activation_func, input_channel, \
                 kernel_sizes, same_padding, num_char_encoding_size, output_size, word_length, grucrf_hidden_size, \
                 i, True, tags, i, pos_size, FCN, 0.6)

            optimizer = optim.Adam(NER.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=True)
            my_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)

            print(device)
            NER.to(device)
            best_score = 0
            best_mat = np.zeros((len(tags)-1,3))
            cnt_idle = 0
            for epoch in range(num_epoch):
                ttt = time()
                print(f'epoch {epoch}')
                all_loss = []
                for ind, batch_x in enumerate(train_loader):
                    print(f'progress: {(100*(grucrf_dropout.index(i)+1)*(max_size_char.index(size_char)+1)*(IND+1))/total_search}')
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
                print(f'total epoch time: {ttt-time()}')

            with open(file_name + '.txt', 'a', encoding='utf8') as f:
                f.write(f'cnt_mat\n')
                f.write(f'I => : {cnt_mat[0,0]}, : {cnt_mat[0,1]}, : {cnt_mat[0,2]}\n')
                f.write(f'B => : {cnt_mat[1,0]}, : {cnt_mat[1,1]}, : {cnt_mat[1,2]}\n')
                f.write(f'O => : {cnt_mat[2,0]}, : {cnt_mat[2,1]}, : {cnt_mat[2,2]}\n')
                f.write(f'best_score: {best_score}\n')
                f.write(f'I => recall: {best_mat[0,0]}, precision: {best_mat[0,1]}, , f1: {best_mat[0,2]}\n')
                f.write(f'B => recall: {best_mat[1,0]}, precision: {best_mat[1,1]}, , f1: {best_mat[1,2]}\n')
                f.write(f'O => recall: {best_mat[2,0]}, precision: {best_mat[2,1]}, , f1: {best_mat[2,2]}\n')
                f.write(f'----------------------------------\n')