# coding: utf-8

import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

def skimming_dataset(documents, vocab):
    # might be better to use np.settdiff1d
    return [[w for w in words if w in vocab] for words in documents]

class GraphBuilder(object):

    def __init__(self, dataset, word_embeddings_dim=300, split=0.1):
        self.dataset = dataset
        self.word_embeddings_dim = word_embeddings_dim
        self.split = split
        self.doc_train_list = []
        self.doc_test_list = []
        self.doc_name_list = []
        self.doc_words_list = []
        self.test_ids = []
        self.train_ids = []
        self.vocab = []
        self.word_id_map = {}
        self.word_doc_freq = {}
        self.label_list = []
        self.word_vector_map = {}
        self.word_vectors = None
        self.word_window_freq = {}
        self.word_pair_count = {}
        self.num_window = 0
        self.y = None
        self.x = None
        self.tx = None
        self.ty = None
        self.allx = None
        self.ally = None

    def load_data(self):
        with open('data/' + self.dataset + '.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.doc_name_list.append(line.strip())
                temp = line.split("\t")
                if temp[1].find('test') != -1:
                    self.doc_test_list.append(line.strip())
                elif temp[1].find('train') != -1:
                    self.doc_train_list.append(line.strip())
    
        
        with open('data/corpus/' + self.dataset + '.clean.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.doc_words_list.append(line.strip())

    def _compute_ids(self, shuffle=True):
        
        for train_name in self.doc_train_list:
            train_id = self.doc_name_list.index(train_name)
            self.train_ids.append(train_id)
        
        for test_name in self.doc_test_list:
            test_id = self.doc_name_list.index(test_name)
            self.test_ids.append(test_id)

        if shuffle:
            random.seed(42)
            random.shuffle(self.train_ids)
            random.shuffle(self.test_ids)
            ids = self.train_ids + self.test_ids

            self.doc_name_list = [self.doc_name_list[i] for i in ids]
            self.doc_words_list = [self.doc_words_list[i] for i in ids]

        # ids = train_ids + test_ids

    def _build_vocab(self):
        word_freq = {}
        word_set = set()
        for doc_words in self.doc_words_list:
            words = doc_words.split()
            for word in words:
                word_set.add(word)
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        
        self.vocab = sorted(list(word_set))

    def _create_word_doc_freq(self):
        word_doc_list = {}

        for i in range(len(self.doc_words_list)):
            doc_words = self.doc_words_list[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    word_doc_list[word].append(i)
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)

        for word, doc_list in word_doc_list.items():
            self.word_doc_freq[word] = len(doc_list)

    def _create_word_id_map(self):
        for i in range(len(self.vocab)):
            self.word_id_map[self.vocab[i]] = i

    def _create_label_list(self):
        # label list
        label_set = set()
        for doc_meta in self.doc_name_list:
            temp = doc_meta.split('\t')
            label_set.add(temp[2])
        self.label_list = sorted(list(label_set))

    def get_train_size(self):
        train_size = len(self.train_ids)
        val_size = int(self.split * train_size)
        train_size = train_size - val_size
        return train_size
    
    def _x_feature(self):

        # different training rates
        # train_size = self.get_train_size()

        # row_x = []
        # col_x = []
        # data_x = []
        # for i in range(train_size):
        #     doc_vec = np.zeros(self.word_embeddings_dim)
        #     doc_words = self.doc_words_list[i]
        #     words = doc_words.split()
        #     doc_len = len(words)
            
        #     # If there is no word vecotr map, useless (no pretrained embeddings)
        #     for word in words:
        #         if word in self.word_vector_map:
        #             word_vector = self.word_vector_map[word]

        #             doc_vec = doc_vec + np.array(word_vector)

        #     for j in range(self.word_embeddings_dim):
        #         row_x.append(i)
        #         col_x.append(j)
        #         # np.random.uniform(-0.25, 0.25)
        #         data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len
        
        self.x = sp.csr_matrix((self.get_train_size(), self.word_embeddings_dim), dtype=np.float32)
        # self.x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        #     train_size, self.word_embeddings_dim))
    
        # We init the features at random

    def _y_feature(self):
        # to one hot, better to use get_dummies from pandas
        y = []
        train_size = self.get_train_size()
        for i in range(train_size):
            doc_meta = self.doc_name_list[i]
            temp = doc_meta.split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            y.append(one_hot)
        self.y = np.array(y)

    def _tx_feature(self):
        # to refactor with _x_feature
        test_size = len(self.test_ids)
        # train_size = len(self.train_ids)
        # row_tx = []
        # col_tx = []
        # data_tx = []
        # for i in range(test_size):
        #     doc_vec = np.zeros(self.word_embeddings_dim)
        #     doc_words = self.doc_words_list[i + train_size]
        #     words = doc_words.split()
        #     doc_len = len(words)
        #     for word in words:
        #         if word in self.word_vector_map:
        #             word_vector = self.word_vector_map[word]
        #             doc_vec = doc_vec + np.array(word_vector)
            

        #     for j in range(self.word_embeddings_dim):
        #         row_tx.append(i)
        #         col_tx.append(j)
        #         data_tx.append(doc_vec[j] / doc_len) 

        # self.tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
        #                 shape=(test_size, self.word_embeddings_dim))
        
        # self.tx = sp.csr_matrix((test_size, self.word_embeddings_dim), dtype=np.float32)
        
        # we init at random
        tx = np.random.uniform(-0.2, 0.2, ((test_size, self.word_embeddings_dim)))
        self.tx = sp.csr_matrix(tx)

    def _ty_feature(self):
        # to refactor with _y_feature
        test_size = len(self.test_ids)
        train_size = len(self.train_ids)
        ty = []
        for i in range(test_size):
            doc_meta = self.doc_name_list[i + train_size]
            temp = doc_meta.split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            ty.append(one_hot)
        self.ty = np.array(ty)
    
    def _create_word_vectors(self):
        np.random.seed(42)
        self.word_vectors = np.random.uniform(-0.01, 0.01,
                            (len(self.vocab), self.word_embeddings_dim))
        for i, word in enumerate(self.vocab):
            if word in self.word_vector_map:
                vector = self.word_vector_map[word]
                self.word_vectors[i] = vector


    def _allx_feature(self):
        
        train_size = len(self.train_ids)
        # # to refactor with _x_feature
        # row_allx = []
        # col_allx = []
        # data_allx = []

        # for i in range(len(self.train_ids)):
        #     doc_vec = np.zeros(self.word_embeddings_dim)
        #     doc_words = self.doc_words_list[i]
        #     words = doc_words.split()
        #     doc_len = len(words)
        #     for word in words:
        #         if word in self.word_vector_map:
        #             word_vector = self.word_vector_map[word]
        #             doc_vec = doc_vec + np.array(word_vector)

        #     for j in range(self.word_embeddings_dim):
        #         row_allx.append(int(i))
        #         col_allx.append(j)
        #         # np.random.uniform(-0.25, 0.25)
        #         data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
        
        # vocab_size = len(self.vocab)
        
        # for i in range(vocab_size):
        #     for j in range(self.word_embeddings_dim):
        #         row_allx.append(int(i + train_size))
        #         col_allx.append(j)
        #         data_allx.append(self.word_vectors.item((i, j)))

        # row_allx = np.array(row_allx)
        # col_allx = np.array(col_allx)
        # data_allx = np.array(data_allx)

        # we don't use embeddings, so just initialize an array of the right size : 
        # allx = np.zeros((train_size, self.word_embeddings_dim), dtype=np.float32)
        # allx = np.append(allx, self.word_vectors, axis=0)
        # self.allx = sp.csr_matrix(allx)

        allx = np.random.uniform(-0.2, 0.2, (train_size, self.word_embeddings_dim))
        allx = np.append(allx, self.word_vectors, axis=0)
        self.allx = sp.csr_matrix(allx)

    def _ally_feature(self):
        ally = []
        train_size = len(self.train_ids)

        for i in range(train_size):
            doc_meta = self.doc_name_list[i]
            temp = doc_meta.split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            ally.append(one_hot)

        for i in range(len(self.vocab)):
            one_hot = [0 for l in range(len(self.label_list))]
            ally.append(one_hot)

        self.ally = np.array(ally)

    def _compute_sliding_windows(self):
        window_size = 20
        windows = []

        for doc_words in self.doc_words_list:
            words = doc_words.split()
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                # print(length, length - window_size + 1)
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)
        self.num_window = len(windows)

        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in self.word_window_freq:
                    self.word_window_freq[window[i]] += 1
                else:
                    self.word_window_freq[window[i]] = 1
                appeared.add(window[i])

        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = self.word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = self.word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in self.word_pair_count:
                        self.word_pair_count[word_pair_str] += 1
                    else:
                        self.word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in self.word_pair_count:
                        self.word_pair_count[word_pair_str] += 1
                    else:
                        self.word_pair_count[word_pair_str] = 1 
        
    def _adjacence_matrix(self):
        row = []
        col = []
        weight = []
        train_size = len(self.train_ids)
        test_size = len(self.test_ids)
        vocab_size = len(self.vocab)

        # pmi as weights
        for key in self.word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = self.word_pair_count[key]
            word_freq_i = self.word_window_freq[self.vocab[i]]
            word_freq_j = self.word_window_freq[self.vocab[j]]
            pmi = log((1.0 * count / self.num_window) /
                    (1.0 * word_freq_i * word_freq_j/(self.num_window * self.num_window)))
            if pmi <= 0:
                continue
            row.append(train_size + i)
            col.append(train_size + j)
            weight.append(pmi)

        # word vector cosine similarity as weights
        doc_word_freq = {}

        for doc_id in range(len(self.doc_words_list)):
            doc_words = self.doc_words_list[doc_id]
            words = doc_words.split()
            for word in words:
                word_id = self.word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1

        for i in range(len(self.doc_words_list)):
            doc_words = self.doc_words_list[i]
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = self.word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                if i < train_size:
                    row.append(i)
                else:
                    row.append(i + vocab_size)
                col.append(train_size + j)
                idf = log(1.0 * len(self.doc_words_list) /
                        self.word_doc_freq[self.vocab[j]])
                weight.append(freq * idf)
                doc_word_set.add(word)

        node_size = train_size + vocab_size + test_size
        self.adj = sp.csr_matrix(
            (weight, (row, col)), shape=(node_size, node_size))

    def _preprocessing(self):
        # preprocessing
        self._compute_ids()
        self._build_vocab()
        self._create_word_doc_freq()
        self._create_word_id_map()
        self._create_label_list()
        self._create_word_vectors()
        self._compute_sliding_windows()

    def _create_features(self):
        # compute features
        self._x_feature()
        self._y_feature()
        self._allx_feature()
        self._ally_feature()
        self._tx_feature()
        self._ty_feature()
        self._adjacence_matrix()

    def build_graph(self):
        self.load_data()
        self._preprocessing()
        self._create_features()

    def dump(self, path='.'):
        with open("{}/ind.{}.x".format(path, self.dataset), 'wb') as f:
            pkl.dump(self.x, f)

        with open("{}/ind.{}.y".format(path, self.dataset), 'wb') as f:
            pkl.dump(self.y, f)

        with open("{}/ind.{}.tx".format(path, self.dataset), 'wb') as f:
            pkl.dump(self.tx, f)

        with open("{}/ind.{}.ty".format(path, self.dataset), 'wb') as f:
            pkl.dump(self.ty, f)

        with open("{}/ind.{}.allx".format(path, self.dataset), 'wb') as f:
            pkl.dump(self.allx, f)

        with open("{}/ind.{}.ally".format(path, self.dataset), 'wb') as f:
            pkl.dump(self.ally, f)

        with open("{}/ind.{}.adj".format(path, self.dataset), 'wb') as f:
            pkl.dump(self.adj, f)

    
    def _evaluating_features(self, X, y):
        """Convert 
        
        Args:
            X (list): a list of strings
            y (list): the associated labels
        """
        # we remove unknown words
        # might be better to use embeddings ?
        skimmed_X = [[w for w in words if w in self.vocab] for words in X]
        # now we want to build the right features : 

        # first, as we just want to evaluate, a X_val and a y_val
        

if __name__ == "__main__":
    g = GraphBuilder('rr', word_embeddings_dim=200)
    g.build_graph()
    g.dump('./data')