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


class GraphBuilder(object):

    def __init__(self, dataset, word_embeddings_dim, split=0.1):
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
        
        self.vocab = list(word_set)

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

    def create_label_list(self):
        # label list
        label_set = set()
        for doc_meta in self.doc_name_list:
            temp = doc_meta.split('\t')
            label_set.add(temp[2])
        self.label_list = list(label_set)

    def get_train_size(self):
        train_size = len(self.train_ids)
        val_size = int(self.split * train_size)
        train_size = train_size - val_size
        return train_size
    
    def _x_feature(self, unsupervised_split=0.1):

        # different training rates
        train_size = self.get_train_size()

        row_x = []
        col_x = []
        data_x = []
        for i in range(train_size):
            doc_vec = np.zeros(self.word_embeddings_dim)
            doc_words = self.doc_words_list[i]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]

                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(self.word_embeddings_dim):
                row_x.append(i)
                col_x.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

        # x = sp.csr_matrix((train_size, word_embeddings_dim), dtype=np.float32)
        self.x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
            train_size, self.word_embeddings_dim))
    
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
        train_size = len(self.train_ids)
        row_tx = []
        col_tx = []
        data_tx = []
        for i in range(test_size):
            doc_vec = np.zeros(self.word_embeddings_dim)
            doc_words = self.doc_words_list[i + train_size]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)
            from pprint import pprint

            for j in range(self.word_embeddings_dim):
                row_tx.append(i)
                col_tx.append(j)
                data_tx.append(doc_vec[j] / doc_len) 

        self.tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                        shape=(test_size, self.word_embeddings_dim))
    
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
        self.word_vectors = np.random.uniform(-0.01, 0.01,
                            (len(self.vocab), self.word_embeddings_dim))
        for i, word in enumerate(self.vocab):
            if word in self.word_vector_map:
                vector = self.word_vector_map[word]
                self.word_vectors[i] = vector


    def _allx_feature(self):
        
        train_size = len(self.train_ids)
        # to refactor with _x_feature
        row_allx = []
        col_allx = []
        data_allx = []

        for i in range(len(self.train_ids)):
            doc_vec = np.zeros(self.word_embeddings_dim)
            doc_words = self.doc_words_list[i]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(self.word_embeddings_dim):
                row_allx.append(int(i))
                col_allx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
        
        vocab_size = len(self.vocab)
        
        for i in range(vocab_size):
            for j in range(self.word_embeddings_dim):
                row_allx.append(int(i + train_size))
                col_allx.append(j)
                data_allx.append(self.word_vectors.item((i, j)))

        row_allx = np.array(row_allx)
        col_allx = np.array(col_allx)
        data_allx = np.array(data_allx)

        self.allx = sp.csr_matrix(
            (data_allx, 
            (row_allx, col_allx)), 
            shape=(train_size + vocab_size, self.word_embeddings_dim))

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