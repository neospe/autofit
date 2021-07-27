#!/usr/bin/env python

"""
parameter tuning with ray tune
experiment logging with sacred
using multiple gpus.
"""

import os
import re
import gc
import random
import itertools
import subprocess
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from tqdm import tqdm
from time import sleep
from joblib import dump, load
import json

# tokenization
from bpemb import BPEmb
import emoji
import html

# automation
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter


#==================

# PARAMETERS

from config_de import *

# system
DB_ADDR = '127.0.0.1:27017'
DB_NAME = 'datalake'
LOG_WORKERS = False
ray.init(num_cpus=8, num_gpus=2, ignore_reinit_error=True, log_to_driver=LOG_WORKERS)

# experiment
EXPERIMENT_NAME = 'DE-default_model_4-dense100-nospam'
CONFIG = default_model_4  #default_model_3  # siehe config.py
NUM_SAMPLES = 100  # how many configurations to sample = how many trials to run

# TODO: zwecks versionierung in config dict definieren
emb_dim = 300
seq_len = 120
validation_split = 0.2

#==================

# PATHS

BPE_MODEL = '../bpemb/de-50k-300d.model'
BPE_EMBEDDING = '../bpemb/de-50k-300d-vectors.txt'

DATA_ALL = 'data/train.tsv'
DATA_TEST = 'data/test.tsv'


def simple_tsv_reader(path):
    f = open(path,'r', encoding='utf-8')
    data = []
    for line in f:
        entry = {}
        cols = line.split('\t')
        entry['text'] = cols[0]
        entry['label'] = cols[1].rstrip()
        data.append(entry)


def create_embeddings(embedding, documents):
    embeddings = []
    labels = []

    for document in tqdm(documents): 
        text = document['text']
        label = document['label']

        array_of_vecs = embedding.embed(text)        
        list_of_vecs = np.split(array_of_vecs, array_of_vecs.shape[0], axis=0)
        list_of_vecs = [v.flatten() for v in list_of_vecs]
        
        #print('len(list_of_vecs):', len(list_of_vecs))
        #print('list_of_vecs[0].shape:', list_of_vecs[0].shape)
        
        embeddings.append(list_of_vecs)
        
        labels.append(label)
    
    return embeddings, labels


def pad_x_matrix(x_matrix, max_tokens=seq_len):
    for sentenceIdx in range(len(x_matrix)):
        sent = x_matrix[sentenceIdx]
        sentence_vec = np.array(sent, dtype=np.float32)  # np.float16

        #print('len(sent[0]):', len(sent[0]))
        #print('sent[0][0].shape:', sent[0][0].shape)
        #print('sentence_vec.shape:', sentence_vec.shape)

        padding_length = max_tokens - sentence_vec.shape[0]
        if padding_length > 0:
            x_matrix[sentenceIdx] = np.append(sent, np.zeros((padding_length, sentence_vec.shape[1])), axis=0)
        else:
            x_matrix[sentenceIdx] = sent[:max_tokens]

    return np.asarray(x_matrix)


def preprocessing():

    # Load data
    print('* Loading data')
    data = simple_tsv_reader(DATA_ALL)
    data = np.asarray(data)
    random.shuffle(data)
    print('data.shape:', data.shape)
    data_len = data.shape[0]

    # assert that data is parsed correctly
    #no_of_samples = 5
    #data_sample = data[np.random.choice(data.shape[0], no_of_samples, replace=False)]
    #print(data_sample)

    # extra test data
    data_test = simple_tsv_reader(DATA_TEST)
    data_test = np.asarray(data_test)
    random.shuffle(data_test)
    print('data_test.shape:', data_test.shape)
    data_test_len = data_test.shape[0]
    
    # check class distribution over all corpora
    c = Counter([item['label'] for item in data])
    print('class distribution:', c)


    # Lookup embeddings
    #embedding = BPEmb(lang="de", vs=bpe_vs, dim=emb_dim)
    embedding = BPEmb(model_file=BPE_MODEL, emb_file=BPE_EMBEDDING, dim=emb_dim)

    print('* Looking up embeddings')
    data_x, data_y = create_embeddings(embedding, data)

    # extra test data
    data_x_test, data_y_test = create_embeddings(embedding, data_test)

    # Pad the x matrix (and possibly convert to 16bit)
    print('* Padding and saving data')
    fp = np.memmap('data_x.dat', dtype='float32', mode='w+', shape=(data_len, seq_len, emb_dim))
    
    # avoid oom: add padded sequences in batches
    batch_len = 2000
    num_batches = int(data_len / batch_len)
    for i in tqdm(range(num_batches)):
        write_start = batch_len * i
        write_end = batch_len * (i + 1)
        fp[write_start:write_end] = pad_x_matrix(data_x[:batch_len], max_tokens=seq_len)
        data_x = data_x[batch_len:]
        #gc.collect()
    fp[batch_len * num_batches:] = pad_x_matrix(data_x, max_tokens=seq_len)  # add rest

    data_y = np.array(data_y)
    np.save('data_y.npy', data_y)
    dump(data_len, 'data_len.joblib')

    # Pad the extra data
    """
    # simple methode, die evtl. zu viel memory kostet
    data_x_test = pad_x_matrix(data_x_test, max_tokens=seq_len)
    data_y_test = np.array(data_y_test)
    np.save('data_x_test.npy', data_x_test)
    np.save('data_y_test.npy', data_y_test)
    """
    fp_test = np.memmap('data_x_test.dat', dtype='float32', mode='w+', shape=(data_test_len, seq_len, emb_dim))
    
    # avoid oom: add padded sequences in batches
    batch_len = 2000
    num_batches = int(len(data_x_test) / batch_len)
    for i in tqdm(range(num_batches)):
        write_start = batch_len * i
        write_end = batch_len * (i + 1)
        fp_test[write_start:write_end] = pad_x_matrix(data_x_test[:batch_len], max_tokens=seq_len)
        data_x_test = data_x_test[batch_len:]
        #gc.collect()
    if len(data_x_test) > 0:
        fp_test[batch_len * num_batches:] = pad_x_matrix(data_x_test, max_tokens=seq_len)  # add rest

    data_y_test = np.array(data_y_test)
    np.save('data_y_test.npy', data_y_test)
    dump(data_test_len, 'data_test_len.joblib')


# TRAINING

def train_func(config):
    """ Create train func with experiment bound in its scope
    Workaround to calling main directly, sacred experiments are not pickle-able
    """
    
    from nn_de import ex
    
    ex.add_config(config)
    ex.run()


if __name__ == '__main__':

    for path in ['logs', 'saved_models']:
        if not os.path.exists(path):
            os.system('mkdir '+path)


    # PREPROCESSING

    if CONFIG['preprocessing'] == True:
        preprocessing()

    
    # DEFINE SCHEDULER
    """
    sched = AsyncHyperBandScheduler(
        time_attr='training_iteration',
        metric='test_f1',
        mode='max',
        max_t=100,
        grace_period=10)
    """


    # DEFINE SEARCH

    # Use bayesian optimisation with TPE implemented by hyperopt
    search_alg = HyperOptSearch(metric='test_f1', mode='max')

    # We limit concurrent trials to 2 since bayesian optimisation doesn't parallelize very well
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)


    # RUN EXPERIMENT

    tune.run(
        train_func,
        
        name=EXPERIMENT_NAME,
        
        #scheduler=sched,

        search_alg=search_alg,

        #stop={  # stoppt kompletten run()  #stop=stopper: kann auch eine function sein, vgl. https://docs.ray.io/en/latest/tune/user-guide.html?highlight=stdout#stopping-trials
        #    'val_f1_m': 0.50,  #0.90
        #    'training_iteration': 300
        #},
        
        num_samples=NUM_SAMPLES,

        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },

        config=CONFIG
        )
