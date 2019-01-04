#!/usr/bin/python
#-*-coding:utf-8-*-
import pandas
import numpy as np

def _main():
    dataset_name = 'rr'

    with open("./data/{}/train.csv".format(dataset_name)) as train_file:
        df_train = pandas.read_csv(train_file, names=["sentences", "label"])
        df_train = df_train.assign(train='train')

    with open("./data/{}/test.csv".format(dataset_name)) as train_file:
        df_test = pandas.read_csv(train_file, names=["sentences", "label"])
        df_test = df_test.assign(train='test')

    df = df_train.append(df_test, ignore_index=True)

    with open("./data/corpus/{}.txt".format(dataset_name), 'w') as corpus_file:
        np.savetxt(corpus_file, df.values[:,0], fmt='%s')

    with open("./data/{}.txt".format(dataset_name), 'w') as metadata_file:
        df.to_csv(metadata_file, header=None, columns=["train", "label"], sep='\t', mode='a') 

if __name__ == "__main__":
    _main()