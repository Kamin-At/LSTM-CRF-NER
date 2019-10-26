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
from torch.utils.data import Dataset, DataLoader, random_split
#from torchcrf import CRF
from torch.utils.data.sampler import SubsetRandomSampler
import random

from sklearn.metrics import confusion_matrix
#from sklearn_crfsuite import metrics

from RULE import RULEs
from my_stuff import *

BS = 256
tags = {0:'I', 1:'B', 2:'O', 3:'<pad>'}
scheduler_n = 15
word_length = 85
early_stop_n = 5
max_size_char = 6
num_search = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data = MyDataloader('./clean85.txt', './label85.txt', RULEs, \
                    word_length, '|', 'char_vec_dictionary.txt',max_size_char, \
                    './fasttext.th.vec', 300, device)


tr, te = get_indices_random_train_test_split(len(data), 1, 0.2, True)
train_loader = DataLoader(data, batch_size=BS, sampler=tr)
test_loader = DataLoader(data, batch_size=BS, sampler=te)

# NER = over_all_NER(BS,300, max_size_char, num_kernels,True,True,1,kernel_sizes,\
#                    True,word_length,135,attention_in, attention_out, gru_hidden_size, \
#                    gru_dropout, True, tags)
#####
# Batch_size: '(int)',\
#                  num_char_vec_features: '(int)',\
#                  hidden_size: '(int)',\
#                  max_num_char: '(int)',\
#                  dropout_gru_char: '(double)',\
#                  bidirectional_char: '(bool)',\
#                  output_char_embed_size: '(int)',\
#                  size_of_embedding: '(int) size of each word embedding vector',\
#                  num_words: '(int) see in overall_char_embedding', \
#                  gru_hidden_size: '(int) see in gru_crf', \
#                  dropout_gru: '(double) see in gru_crf', \
#                  bidirectional: '(bool)', \
#                  tags: '(dict[int: str]) see in gru_crf', DO_FCN_GRUCRF: '(double)', DOchar_FCN: '(double)')
#####

with open('my_logs.txt', 'w', encoding ='utf8') as f:
    for cur_ind in range(num_search):
        torch.cuda.empty_cache()
        grucrf_dropout = random.uniform(0.1,0.6)#0.5
        gruchar_dropout = random.uniform(0.1,0.6)#0.5
        DO_FCN_GRUCRF = random.uniform(0.1,0.6)#0.5
        DO_FCN_CHAR = random.uniform(0.1,0.6)#0.5
        grucrf_hidden_size = random.choice([8,16,32,64,128])#5
        hidden_size_char_gru = random.choice([8,16,32,64,128])#20
        LR = 5*10**random.uniform(-3,-5)#0.001
        f.write(f'cur_ind: {cur_ind}\n')
        f.write(f'grucrf_dropout: {grucrf_dropout}, gruchar_dropout: {gruchar_dropout}\n')
        f.write(f'DO_FCN_GRUCRF: {DO_FCN_GRUCRF}, DO_FCN_CHAR: {DO_FCN_CHAR}\n')
        f.write(f'grucrf_hidden_size: {grucrf_hidden_size}, hidden_size_char_gru: {hidden_size_char_gru}\n')
        f.write(f'LR: {LR}\n')

        NER = over_all_NER2(BS, 135, hidden_size_char_gru, max_size_char, gruchar_dropout,\
            True, 100, 300, word_length, grucrf_hidden_size, grucrf_dropout, True, \
                tags, DO_FCN_GRUCRF, DO_FCN_CHAR)

        optimizer = optim.Adam(NER.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=True)
        my_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

        print(device)
        NER.to(device)

        best_score = 0
        best_mat = np.zeros((len(tags)-1,3))
        cnt_idle = 0
        for epoch in range(6):
            print(f'epoch {epoch}')
            all_loss = []
            for ind, batch_x in enumerate(train_loader):
                if ind%5 == 0:
                    print(ind)
                if ind%scheduler_n == scheduler_n-1:
                    total_loss = sum(all_loss[-scheduler_n:])/(scheduler_n)
                    my_scheduler.step(total_loss)
                NER = NER.train()
                NER.zero_grad()
                t1 = time()
                loss = NER(batch_x)
                loss = loss*(-1)
                print(f'time per batch: {time() - t1}')
                print(loss)
                all_loss.append(loss)
                loss.backward()
                nn.utils.clip_grad_value_(NER.parameters(), 10)
                optimizer.step()
            total_loss = sum(all_loss)/(ind + 1)
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
        f.write(f'best_score: {best_score}\n')
        f.write(f'best_mat\n')
        f.write(f'I => recall: {best_mat[0,0]}, precision: {best_mat[0,1]}, , f1: {best_mat[0,2]}\n')
        f.write(f'B => recall: {best_mat[1,0]}, precision: {best_mat[1,1]}, , f1: {best_mat[1,2]}\n')
        f.write(f'O => recall: {best_mat[2,0]}, precision: {best_mat[2,1]}, , f1: {best_mat[2,2]}\n')
        f.write(f'best_mat\n')
        f.write(f'----------------------------------\n')