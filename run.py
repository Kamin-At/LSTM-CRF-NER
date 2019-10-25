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
from torchcrf import CRF
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.metrics import confusion_matrix
from sklearn_crfsuite import metrics

from RULE import RULEs
from layes import *

####################################################
################# Configuration ####################
#################################################### 

data_dir = '../clean169_no_stopwords_NE.txt'
label_dir = '../lable169_no_stopwords_NE.txt'
BS = 16
max_size_char = 6
gru_dropout = 0.5
word_length = 169
num_kernels =[2,2]
kernel_sizes = [3,5]
gru_hidden_size = 5
attention_in = 50
attention_out = 100
tags = {0:'I', 1:'B', 2:'O', 3:'<pad>'}
num_epoch = 10

####################################################
################## Setting #########################
#################################################### 

data = MyDataloader(data_dir, label_dir, RULEs, \
                    169, '|', 'char_vec_dictionary.txt',6, \
                    '../fasttext.th.vec', 300)

va, te = get_indices_random_val_test_split(len(data), 1, 0.0001, True)
# tr, te = get_indices_random_train_test_split(len(data), 1, 0.0001, True)
test_loader = DataLoader(data, batch_size=BS, sampler=te)
va_loader = DataLoader(data, batch_size=BS, sampler=va)

NER = over_all_NER(BS,300, max_size_char, num_kernels,True,True,1,kernel_sizes,\
                   True,word_length,135,attention_in, attention_out, gru_hidden_size, \
                   gru_dropout, True, tags)
optimizer = optim.SGD(NER.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.9)
my_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

####################################################
########### After This Is For Training #############
#################################################### 

for epoch in range(num_epoch):
    print(f'epoch {epoch}')
    all_loss = []
    for ind, batch_x in enumerate(va_loader):
        if ind%5 == 0:
            print(ind)
        NER = NER.train()
        NER.zero_grad()
        t1 = time()
        loss = NER(batch_x)
        loss = loss*(-1)
        
        print(f'time per batch: {time() - t1}')
        #print(loss)
        all_loss.append(loss)
        loss.backward()
        nn.utils.clip_grad_value_(NER.parameters(), 10)
        optimizer.step()
    total_loss = sum(all_loss)/(ind+1)
    print(f'total loss of epoch: {total_loss.item()}')
    my_scheduler.step(total_loss)
    print('testing')
    per_mat = np.zeros((len(tags), 3))
    for ind, batch_test in enumerate(test_loader):
        NER = NER.eval()
        output = NER.predict(batch_test)
        per_mat += eval_score(tags, output, batch_test[2])
    per_mat = per_mat/(ind+1)
    print(per_mat)
    print(f'overall score: {sum(per_mat[:,2])/len(tags)}')
    print('--------------------')