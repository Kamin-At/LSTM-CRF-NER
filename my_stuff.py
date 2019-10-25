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

#{0:'I', 1:'B', 2:'O', 3:'<PAD>'}
class MyDataloader(Dataset):
    def __init__(self, TextDir: '.txt extension of samples', LabelDir: '.txt extension of labels',rules:\
                 'the rules to be replaced => see in RULE.py', Len_word_vec: 'size of word vector', \
                delimiter: '(str) delimiter used to separate data', dir_char_dictionary: \
                '(str) see in CharEmbedding', max_len_char: '(int) see in CharEmbedding', \
                fasttext_dictionary_dir: '(str) see in WordEmbedding',\
                Len_embedded_vector: '(int) see in WordEmbedding', device) -> None:
        super().__init__()
        self.DF = pd.read_csv(TextDir, names=['text'])
        self.Label_DF = pd.read_csv(LabelDir, names=['text'])
        self.rules = rules
        self.Len_word_vec = Len_word_vec
        self.delimiter = delimiter
        self.char_embedder = CharEmbedding(dir_char_dictionary, max_len_char)
        self.word_embedder = WordEmbedding(fasttext_dictionary_dir, Len_embedded_vector)
        self.device = device
    def __len__(self):
        return len(self.DF)
    def __getitem__(self, Index) -> '(sample: (torch.tensor), label: (torch.tensor))':
        all_words = [word.strip() for word in self.DF['text'][Index].strip().split(self.delimiter)]
        for i in range(len(all_words)):
            for rule in self.rules:
                all_words[i] = re.sub(*rule, all_words[i])
        Label = [float(word.strip()) for word in self.Label_DF['text'][Index].strip().split(self.delimiter)]
        mask = [1.0]*len(all_words)
        if len(all_words) < self.Len_word_vec:
            Label = Label + [3.0]*(self.Len_word_vec - len(all_words))
            mask = mask + [0.0]*(self.Len_word_vec - len(all_words))
            all_words = all_words + ['<pad>']*(self.Len_word_vec - len(all_words))
        char_embed = self.char_embedder.embed(all_words)
        word_embed = self.word_embedder.embed(all_words)
        
        # print(len(all_words))
        # print(len(Label))
        # print(len(mask))
        # print('----------')
        return (char_embed.to(self.device), word_embed.to(self.device), torch.tensor(Label).to(self.device), torch.tensor(mask).to(self.device), len(all_words))
    

class CharEmbedding():
    def __init__(self,\
    dir_char_dictionary: '(str) .txt',\
    max_len_char: '(int) max size of char representation, for example: given max_len_char=3 and word= "abcde" => only "abc" is used'):
    #Example: given embed_capital=True and 'a' is embedded as array([1.,0.,0.,0.,0]). 'A' is then embedded as array([1.,0.,0.,0.,1.])
        self.dictionary = {}
        self.max_len_char = max_len_char
        with open(dir_char_dictionary, 'r', encoding='utf8') as f:
            for line in f:
                tmp_data = line.strip().split()
                self.dictionary[tmp_data[0]] = np.array([float(Char) for Char in tmp_data[1:]])
    def embed(self, list_of_words: '(list[str]) example: ["ฉัน","กิน","ข้าว"]'):
        #Note: 1 outer list is for 1 word.
        output = []
        for word in list_of_words:
            embedded_word = []
            tmp_word = word
            if len(word) > self.max_len_char:
                tmp_word = tmp_word[:self.max_len_char]
            for Char in tmp_word:
                if Char in self.dictionary:
                    tmp_vector = self.dictionary[Char]
                else:
                    tmp_vector = np.zeros(self.dictionary['a'].shape)
                embedded_word.append(tmp_vector)
            if len(embedded_word) < self.max_len_char:
                for i in range(self.max_len_char - len(embedded_word)):
                    embedded_word.append(np.zeros(self.dictionary['a'].shape))
            output.append(torch.tensor(embedded_word))
        return torch.stack(output)

class WordEmbedding():
    #use fasttext embedding ==> read from a file
    def __init__(self, fasttext_dictionary_dir: '(str) .vec extension of words and embedded_vectors',\
     Len_embedded_vector: '(int) size of embedded each vector (300 for fasttext) **Count only numbers not words'\
     ) -> None:
        #example of format in fasttext_dictionary_dir
        #กิน 1.0 -2.666 -3 22.5 .... \n
        #นอน 1.5 -5.666 3 9.5 .... \n
        #...
        #...
        self.dictionary = {}
        self.Len_embedded_vector = Len_embedded_vector
        with open(fasttext_dictionary_dir, 'r', encoding = 'utf8') as f:
            for line in f:
                tmp_line = line.strip()
                tmp_words = tmp_line.split()
                if tmp_line != '' and len(tmp_words) == self.Len_embedded_vector + 1:
                    self.dictionary[tmp_words[0]] = np.array([float(element) for element in tmp_words[1:]])
                else:
                    continue
    def embed(self, list_of_words: '(List[str]) for example: ["ฉัน","กิน","ข้าว"]'):
        tmp_list = []
        for word in list_of_words:
            if word in self.dictionary:
                tmp_list.append(self.dictionary[word])
            else:
                #in case of OOV: Zero-vector is used.
                tmp_list.append(np.zeros(self.Len_embedded_vector))
        return torch.tensor(tmp_list)

############### RNN encoding ######################
class RNN_char(nn.Module):
    def __init__(self, num_char_vec_features, hidden_size, num_layers, dropout_gru, bidirectional, \
                output_size, dropout_FCN, num_word):
        super().__init__()
        self.gru = nn.GRU(input_size=num_char_vec_features, hidden_size=hidden_size, num_layers=num_layers,\
                          batch_first = True, dropout=dropout_gru, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size*2*num_layers, output_size)
        self.BN = nn.BatchNorm1d(num_word)
        self.dropout = nn.Dropout(dropout_FCN)
        self.num_layers = num_layers
    def forward(self, x):
        batch_size, word_seq, char_seq, char_vec = x.size()
        tmp_list = []
        for i in range(word_seq):
            tmp_compute , _ = self.gru(x[:,i,:,:].float())
            tmp_list.append(tmp_compute.contiguous().view(batch_size,-1))
        tmp_compute = torch.stack(tmp_list,1)
        #print(tmp_compute.size())
        tmp_compute = self.dropout(tmp_compute)
        tmp_compute = self.linear(tmp_compute)
        #print(tmp_compute.size())
        tmp_compute = F.relu(self.BN(tmp_compute))#>>linear >> BachNorm >> relu
        return tmp_compute
    
class over_all_NER2(nn.Module):
    def __init__(self, Batch_size: '(int)',\
                 num_char_vec_features: '(int)',\
                 hidden_size: '(int)',\
                 max_num_char: '(int)',\
                 dropout_gru_char: '(double)',\
                 bidirectional_char: '(bool)',\
                 output_char_embed_size: '(int)',\
                 size_of_embedding: '(int) size of each word embedding vector',\
                 num_words: '(int) see in overall_char_embedding', \
                 gru_hidden_size: '(int) see in gru_crf', \
                 dropout_gru: '(double) see in gru_crf', \
                 bidirectional: '(bool)', \
                 tags: '(dict[int: str]) see in gru_crf', DO_FCN_GRUCRF: '(double)', DOchar_FCN: '(double)'):
        super().__init__()
        self.gru_char = RNN_char(num_char_vec_features, hidden_size, max_num_char, dropout_gru_char, \
                                 bidirectional_char, output_char_embed_size, DOchar_FCN, num_words)
        self.gru_crf_layer = gru_crf(size_of_embedding + output_char_embed_size, \
                                     gru_hidden_size, num_words, dropout_gru, bidirectional, tags, DO_FCN_GRUCRF)
    def forward(self, x):
        tmp_compute = self.gru_char(x[0])
        #print(tmp_compute.size())
        #print(x[1].size())
        tmp_compute = torch.cat([tmp_compute, x[1].float()], 2)
        #print(tmp_compute.size())
        tmp_gru_crf = self.gru_crf_layer((tmp_compute, x[4]), x[2], x[3].long())
        return tmp_gru_crf
    def predict(self, x):
        tmp_compute = self.gru_char(x[0])
        tmp_compute = torch.cat([tmp_compute, x[1].float()], 2)
        tmp_gru_crf = self.gru_crf_layer.predict((tmp_compute, x[4]), x[3].long())
        return tmp_gru_crf


# class TimeDistributed(nn.Module):
#     def __init__(self, layer: '(nn.Module) layer to be processed', time_steps: '(int)'):
#         super().__init__()
#         self.layers = nn.ModuleList([layer for i in range(time_steps)])

#     def forward(self, x) -> '(torch.tensor) shape=(1, embedding_size)':
#         batch_size, time_steps, C, H, W = x.size()
#         output = torch.tensor([])
#         for i in range(time_steps):
#           output_t = self.layers[i](x[:, i, :, :, :])
#           output_t  = torch.flatten(output_t)
#           output = torch.cat((output, output_t ), 1)
#         return output

# class Convs(nn.Module):
#     def __init__(self, List_of_kernel_sizes: 'example: [(3,100),(5,100),(7,100)]', List_num_filter: 'example: \
#     [64,64,128] ***len(List_num_filter) must equal to len(List_of_kernel_sizes)***',\
#     use_BN: 'see My2DConv', activation_func: 'see My2DConv', input_channel: 'see My2DConv', \
#     same_padding: 'see My2DConv', time_steps: 'see TimeDistributed'):
#         tmp_List_layers = []
#         for ind, kernel_size in enumerate(List_of_kernel_sizes):
#             tmp_List_layers.append(TimeDistributed(My2DConv(List_num_filter[ind], use_BN, \
#             activation_func, input_channel, kernel_size, same_padding), time_steps))
#         self.Layer_list = nn.ModuleList(tmp_List_layers)

def get_index(len_row, len_col)->'(iterator of all ((int)row, (int)col))':
    for i in range(len_row):
        for j in range(len_col):
            yield(i,j)

def get_longest_seq_len(MASK: '(torch.tensor: shape=(batch_size, num_words)) \
    of mask 1 for non padding, 0 for otherwise')->'(int) col index of first zero in\
    of the longest sequence example: x=torch.tensor([[1,1,0],[1,0,0]]) -> return 2':
    tmp_mask = MASK.numpy()
    if len(tmp_mask.shape) != 1:
        tmp_mask = np.sum(tmp_mask,0)
    col = 0
    for i in range(tmp_mask.shape[0]):
        if tmp_mask[i]==0:
            col = i
            break
    if col == 0:
        col = tmp_mask.shape[0]
    return col

class overall_char_embedding(nn.Module):
    def __init__(self, output_size: '(tuple of ints): (batch_size, embedding_size_per_word)',
    max_len_char: '(int) see in CharEmbedding',\
    nums_filter: '(list) list of number of filters according to each kernel_sizes (respectively)',
    use_BN: 'see in My2DConv',
    activation_func: 'see in My2DConv',
    input_channel: 'see in My2DConv',
    kernel_sizes: '(list[int]) list of size of kernels used, and they will be computed concurrently',
    same_padding: 'see in My2DConv',
    num_words: 'number of words used in 1 sample',
    num_char_encoding_size: 'size of encoding for each char'):
        super().__init__()
        self.batch_size, self.embedding_size_per_word = output_size
        tmp_cnn_models = []
        for ind_cnn, kernel_size in enumerate(kernel_sizes):
            tmp_cnn_models.append(\
            My2DConv(nums_filter[ind_cnn], use_BN, activation_func, input_channel,\
            (kernel_size, num_char_encoding_size), same_padding)
            )
        self.num_words = num_words
        self.CNNs = nn.ModuleList(tmp_cnn_models)
        self.MyMaxPool = nn.MaxPool2d((1, num_char_encoding_size), stride= (1,1))
        self.MyFCN = nn.Linear(sum(nums_filter)*max_len_char, output_size[1])
    def forward(self, x):
        batch_size, num_word, num_char, embedding_size = x.size()
        #print(x.size())
        tmp_compute = x.view(batch_size, num_word, 1, num_char, \
        embedding_size)
        all_output_list = []
        for num_word in range(self.num_words):
            tmp_output_cnn = []
            for tmp_cnn in self.CNNs:
                tmp_output_cnn.append(self.MyMaxPool(tmp_cnn(tmp_compute[:,\
                num_word,:,:,:])).view((batch_size, -1)))
            all_output_list.append(F.relu(self.MyFCN(torch.cat(tmp_output_cnn, 1))))
        #print(all_output_list[0].size())
        #print(len(all_output_list))
        all_output_list = torch.stack(all_output_list, dim=1)
        return all_output_list
                
class gru_crf(nn.Module):
    def __init__(self, num_input_features: '(int) number of input features', hidden_size: '(int) number of\
    hidden features the outputs will also have hidden_size features', num_layers: '(int) number of \
    recursion', dropout_gru, bidirectional: '(bool) if True, use bidirectional GRU',\
    tags: "(dict[int: str])example: {0:'I', 1:'B', 2:'O', 3:'<PAD>'}", dropout_FCN: '(double)'):
        super().__init__()
        self.gru = nn.GRU(input_size=num_input_features, hidden_size=hidden_size, num_layers=num_layers,\
        batch_first = True, dropout=dropout_gru, bidirectional=bidirectional)
        #all_transition=allowed_transitions('IOB1', tags)
        #self.crf = CRF(num_tags=len(tags), batch_first= True)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, len(tags))
        self.crf = ConditionalRandomField(len(tags))
        self.dropout = nn.Dropout(dropout_FCN)
        
    def forward(self, samples, target: '(torch.tensor) shape=(...............,)the target tags to be used',\
                mask: 'True for non-pad elements'):
        length = samples[1]
        samples = samples[0]
        batch_size, words, _ = samples.size()
        tmp_t = time()
        tmp_compute = self.gru(samples)[0].view(batch_size, words, -1)
#         print(f'total GRU time: {time() - tmp_t}')
        index_to_cut = max(length).item()#get_longest_seq_len(mask)
        #length = torch.mean(length.float()).item()
        ##############################################
        ###cut padding some parts out#################
        tmp_compute = tmp_compute[:, :index_to_cut,:]
        target = target[:, :index_to_cut]
        mask = mask[:, :index_to_cut]
        #print(tmp_compute.size())
        tmp_compute = self.dropout(tmp_compute)
        tmp_compute = F.relu(self.linear(tmp_compute))
        tmp_compute = self.dropout(tmp_compute)
        tmp_compute = F.relu(self.linear2(tmp_compute))
        #print(tmp_compute.size())
        nll_loss = self.crf(tmp_compute,target.long(),mask)
#         print(f'total CRF time: {time() - tmp_t}')
        return nll_loss#/length
    def predict(self, samples, mask):
        length = samples[1]
        samples = samples[0]
        batch_size, words, _ = samples.size()
        tmp_t = time()
        tmp_compute = self.gru(samples)[0].view(batch_size, words, -1)
#         print(f'total GRU time: {time() - tmp_t}')
        index_to_cut = max(length).item()#get_longest_seq_len(mask)
        ##############################################
        ###cut padding some parts out#################
        tmp_compute = tmp_compute[:, :index_to_cut,:]
        mask = mask[:, :index_to_cut]
        #print(tmp_compute.size())
        
        tmp_compute = F.relu(self.linear(tmp_compute))
        tmp_compute = F.relu(self.linear2(tmp_compute))
        #print(tmp_compute.size())
        tmp_t = time()
        tmp_tags = self.crf.viterbi_tags(tmp_compute,mask)
#         print(f'total CRF prediction time: {time() - tmp_t}')
        return tmp_tags
    
class My2DConv(nn.Module):
    def __init__(self, num_filter: '(int) number of filters', use_BN: '(bool) if True, use 2d-batchnorm after linear conv',\
    activation_func: '(bool) if True, use RELU after BN', input_channel: '(int) number of input channels', \
    kernel_size: '(tuple): (width, height) size of the kernels', same_padding: '(bool) if True, input_w,input_h=output_w,output_h'):
        super().__init__()
        if same_padding:
            #assume that dialation = 1 and stride = 1
            self.padding = (math.floor((kernel_size[0] - 1)/2), math.floor((kernel_size[1] -1)/2))
        else:
            self.padding = 0
        self.Conv = nn.Conv2d(input_channel, num_filter, kernel_size, padding= self.padding)
        self.use_BN = use_BN
        self.activation_func = activation_func
        if self.use_BN:
            self.BN = nn.BatchNorm2d(num_filter)

    def forward(self, input_data: '(torch.tensor) dimension= (batch_size, num_channel_in, in_height, in_width)') \
    -> '(torch.tensor) shape= (batch_size, num_filter, in_height, in_width)':
        tmp_compute = self.Conv(input_data.float())
        if self.use_BN:
            tmp_compute = self.BN(tmp_compute)
        if self.activation_func:
            tmp_compute = nn.ReLU()(tmp_compute)
        return tmp_compute
        



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class AttentionBetweenWordsAndChars(nn.Module):
    def __init__(self, hidden_size: '(int) size of key, query and value vectors',\
    input_vec_size: '(int) incase of fasttext input_vec_size=300'):
        super().__init__()
        self.K_FCN = nn.Linear(input_vec_size, hidden_size)
        self.Q_FCN = nn.Linear(input_vec_size, hidden_size)
        self.V_FCN = nn.Linear(input_vec_size, hidden_size)
        self.AttLayer = ScaledDotProductAttention(math.sqrt(hidden_size), 0.1)
    def forward(self, char_vectors, word_vectors):
        batch_size, word_size, _ = word_vectors.size()
        word_vectors = word_vectors.float()
        char_vectors = char_vectors.float()
#         print(word_vectors.size())
#         print(char_vectors.size())
        K = torch.stack([self.K_FCN(word_vectors),self.K_FCN(char_vectors)],dim = 2)
        Q = torch.stack([self.Q_FCN(word_vectors),self.Q_FCN(char_vectors)],dim = 2)
        V = torch.stack([self.V_FCN(word_vectors),self.V_FCN(char_vectors)],dim = 2)
        all_output_list = []
        for word_ind in range(word_size):
            all_output_list.append(self.AttLayer(Q[:,word_ind,:,:], \
            K[:,word_ind,:,:], V[:,word_ind,:,:])[0].view(batch_size,-1))

        return torch.stack(all_output_list,dim = 1)
    
class over_all_NER(nn.Module):
    def __init__(self, Batch_size: '(int)',\
                 size_of_embedding: '(int) size of each word embedding vector',\
                 max_len_char: '(int) see overall_char_embedding',\
                 num_conv_filters: '(list[int]) see in overall_char_embedding', \
                 use_BN: '(bool) see in overall_char_embedding', \
                 use_activation: '(bool) see in overall_char_embedding', \
                 num_conv_input_channel: '(int) see in overall_char_embedding', \
                 kernel_sizes: '(list[tuple[int, int]]) see in overall_char_embedding', \
                 use_same_padding: '(bool) see in overall_char_embedding', \
                 num_words: '(int) see in overall_char_embedding', \
                 num_char_encoding_size: '(int) see in overall_char_embedding', \
                 att_hidden_size: '(int) see in AttentionBetweenWordsAndChars', \
                 num_input_features: '(int) see in gru_crf', gru_hidden_size: '(int) see in gru_crf', \
                 dropout_gru: '(double) see in gru_crf', bidirectional: '(bool)', \
                 tags: '(dict[int: str]) see in gru_crf'):
        super().__init__()
        self.char_embed = overall_char_embedding((Batch_size,size_of_embedding), max_len_char, num_conv_filters, \
                                                 use_BN, use_activation, num_conv_input_channel, kernel_sizes, \
                                                 use_same_padding, num_words, num_char_encoding_size)
        self.my_attention = AttentionBetweenWordsAndChars(att_hidden_size, size_of_embedding)
        self.gru_crf_layer = gru_crf(num_input_features, gru_hidden_size, num_words, dropout_gru, \
                                bidirectional, tags)
        self.Batch_size = Batch_size
    def forward(self, x):
        tmp_compute = self.char_embed(x[0])
        tmp_att = self.my_attention(tmp_compute, x[1])
        tmp_gru_crf = self.gru_crf_layer(tmp_att, x[2], x[3].long())
        return tmp_gru_crf#/self.Batch_size
    def predict(self, x):
        tmp_compute = self.char_embed(x[0])
        tmp_att = self.my_attention(tmp_compute, x[1])
        tmp_tags = self.gru_crf_layer.predict(tmp_att, x[3].long())
        return tmp_tags

def get_indices_random_train_test_split(dataset_size:'(int) number of rows', random_seed: '(int)',\
                                        validation_split: '(double)', shuffle_dataset: '(bool)'):
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler
  
def get_indices_random_val_test_split(dataset_size:'(int) number of rows', random_seed: '(int)',\
                                        validation_split: '(double)', shuffle_dataset: '(bool)'):
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    test_indices, val_indices = indices[split: 2*split], indices[:split]
    # Creating PT data samplers and loaders:
    test_sampler = SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return test_sampler, valid_sampler

def eval_score(tags: '(dict[int: str])', pred: '(list[(list, float)])', label: 'torch.tensor'):
    pred = np.array([np.array(i[0]) for i in pred])
    label = label.cpu().numpy().astype('int8')
    label = [label[i][:len(pred[i])] for i in range(len(pred))]
    conf_mat = np.zeros((len(tags), len(tags)))
#     print(len(label))
#     print(len(pred))
#     print('---------------')
    for i in range(len(label)):
#         print(len(label[i]))
#         print(len(pred[i]))
        conf_mat += confusion_matrix(label[i],pred[i],range(len(tags)))
    performance_mat = np.zeros((len(tags), 3))#recall, precision, f1-score
    for i in range(len(tags)):
        if np.sum(conf_mat[i]) == 0:
            performance_mat[i][0] = 0
        else:
            performance_mat[i][0] = conf_mat[i][i]/np.sum(conf_mat[i])
        if np.sum(conf_mat[:,i]) == 0:
            performance_mat[i][1] = 0
        else:
            performance_mat[i][1] = conf_mat[i][i]/np.sum(conf_mat[:,i])
        if performance_mat[i][1]+performance_mat[i][0] == 0:
            performance_mat[i][2] = 0
        else:
            performance_mat[i][2] = (2*performance_mat[i][0]*performance_mat[i][1])/(performance_mat[i][1]+performance_mat[i][0])
    return performance_mat