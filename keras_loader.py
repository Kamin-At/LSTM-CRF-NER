import numpy as np
import pandas as pd
import regex as re


class MyDataloader(Dataset):
    def __init__(self, TextDir: '.txt extension of samples', LabelDir: '.txt extension of labels',rules:\
                 'the rules to be replaced => see in RULE.py', Len_word_vec: 'size of word vector', \
                delimiter: '(str) delimiter used to separate data', dir_char_dictionary: \
                '(str) see in CharEmbedding', max_len_char: '(int) see in CharEmbedding', \
                fasttext_dictionary_dir: '(str) see in WordEmbedding',\
                Len_embedded_vector: '(int) see in WordEmbedding', device, POSDir: '(str) .txt extension of POS',\
                POSMapping: 'see in POSMap.py', BS: '(int) batch size') -> None:
        super().__init__()
        self.DF = pd.read_csv(TextDir, names=['text'])
        self.Label_DF = pd.read_csv(LabelDir, names=['text'])
        self.pos_DF = pd.read_csv(POSDir, names=['text'])
        self.rules = rules
        self.Len_word_vec = Len_word_vec
        self.delimiter = delimiter
        self.char_embedder = CharEmbedding(dir_char_dictionary, max_len_char)
        self.word_embedder = WordEmbedding(fasttext_dictionary_dir, Len_embedded_vector)
        self.device = device
        self.pos_embedder = POSEmbedding(POSMapping)
        self.BS = BS
    def __len__(self):
        return len(self.DF)//self.BS
    def __getitem__(self, Index) -> '(sample: (torch.tensor), label: (torch.tensor))':
        all_words = [word.strip() for word in self.DF['text'][Index].strip().split(self.delimiter)]
        for i in range(len(all_words)):
            for rule in self.rules:
                all_words[i] = re.sub(*rule, all_words[i])
        Label = [float(word.strip()) for word in self.Label_DF['text'][Index].strip().split(self.delimiter)]
        mask = [1.0]*len(all_words)
        POS = [pos.strip() for pos in self.pos_DF['text'][Index].strip().split(self.delimiter)]
        tmp_length = len(all_words)
        if len(all_words) < self.Len_word_vec:
            Label = Label + [3.0]*(self.Len_word_vec - len(all_words))
            mask = mask + [0.0]*(self.Len_word_vec - len(all_words))
            POS = POS + ['<pad>']*(self.Len_word_vec - len(all_words))
            all_words = all_words + ['<pad>']*(self.Len_word_vec - len(all_words))
        char_embed = self.char_embedder.embed(all_words)
        word_embed = self.word_embedder.embed(all_words)
        pos_embed = self.pos_embedder.embed(POS)
        # print(len(all_words))
        # print(len(Label))
        # print(len(mask))
        # print('----------')
        return (char_embed.to(self.device), word_embed.to(self.device), \
                torch.tensor(Label).to(self.device), torch.tensor(mask).to(self.device), \
                tmp_length, pos_embed.float().to(self.device))

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
        return np.array(tmp_list)

class POSEmbedding():
    def __init__(self, POSMapping: 'see in POSMap.py'):
        self.dictionary = POSMapping
        self.size = len(self.dictionary)
    def embed(self, list_of_POSs:'(list[str]) example: ["NOUN","VERB","NOUN"]'):
        tmp_list = []
        for POS in list_of_POSs:
            POS = POS.strip()
            if POS == '<pad>':
                tmp_list.append(np.zeros(self.size))
            else:
                tmp_data = np.zeros(self.size)
                tmp_data[self.dictionary[POS]] = 1
                tmp_list.append(tmp_data)
        return np.array(tmp_list)
