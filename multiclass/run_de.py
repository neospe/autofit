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

# dataload
from dataload.loader import *
from dataload.preproc import *

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
os.system('$env:CUDA_VISIBLE_DEVICES="0,1"')
os.system('$env:TUNE_DISABLE_STRICT_METRIC_CHECKING="1"')
DB_ADDR = '127.0.0.1:27017'
DB_NAME = 'datalake'
LOG_WORKERS = False
ray.init(num_cpus=8, num_gpus=2, ignore_reinit_error=True, log_to_driver=LOG_WORKERS)

# experiment
EXPERIMENT_NAME = 'DE-MULTI-default_4'
CONFIG = default_4  # siehe config.py
NUM_SAMPLES = 10  # how many configurations to sample = how many trials to run

# TODO: zwecks versionierung in config dict definieren
no_of_classes = 5  # multiclass label, e.g. [0, 0, 1, 0, 0]
emb_dim = 300
seq_len = 120
validation_split = 0.2

#==================

# PATHS

BPE_MODEL = '../bpemb/de-50k-300d.model'
BPE_EMBEDDING = '../bpemb/de-50k-300d-vectors.txt'

DATA2018_TRAIN = 'GermEval-2018-Data/germeval2018.training.txt'
DATA2018_TEST = 'GermEval-2018-Data/germeval2018.test.txt'  # 3532 items
DATA2018_EXTRA = 'germeval2018-upinf/labeled_tweets.tsv'  # 3728 items

DATA2019_TRAIN_1 = 'GermEval-2019-Data/germeval2019_training_subtask12.txt'
DATA2019_TRAIN_2 = 'GermEval-2019-Data/germeval2019.training_subtask3.txt'
DATA2019_TEST_1 = 'GermEval-2019-Data/germeval2019GoldLabelsSubtask12.txt'  # 3031 items
DATA2019_TEST_2 = 'GermEval-2019-Data/germeval2019GoldLabelsSubtask3.txt'

GERMAN_SPAM = 'german-spam/german-spam.txt'
DATA_MINITEST = 'eval/test-de.csv'  # loader: local=True

# data gesamt
DATA_ALL = [DATA2018_TRAIN, DATA2018_TEST, DATA2019_TRAIN_1, DATA2019_TEST_1, DATA2019_TRAIN_2, DATA2019_TEST_2, GERMAN_SPAM]

# ANM: extra testdaten müssen in einem eigenen directory liegen, sonst wird es implizit bei den augmentations mit geladen (geht nach dataset/directory name, e.g. 'GermEval-2019-Data') -> evtl. TODO: beim laden der augmentations nochmal mit DATA_TEST abgleichen
DATA_TEST = [DATA2018_EXTRA]

# augmentations
AUG_BACKLOG_PATH = ['backlog.json']
AUGS_TO_USE = [DATA2018_TRAIN, DATA2019_TRAIN_1]
DEBIAS_MADLIBS = 'social-bias/selection/bias_madlibs_89k-German.csv'


def create_embeddings(embedding, documents):
    embeddings = []
    labels = []

    for document in tqdm(documents):
        text = clean_str(document['text'])
        if 'label2' in document.keys():
            label_str = document['label2']
        else:
            label_str = document['label']
        label = [0 for i in range(no_of_classes)]
        label[MAPPING_AGGRESSION_MULTI[label_str]] = 1

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
    data = germeval(files=DATA_ALL)
    data += debias_madlibs(files=[DEBIAS_MADLIBS])
    data += augmentations(files=AUG_BACKLOG_PATH+AUGS_TO_USE)
    data = np.asarray(data)
    random.shuffle(data)
    print('data.shape:', data.shape)
    data_len = data.shape[0]

    # extra test data
    data_test = germeval(files=DATA_TEST)
    data_test = np.asarray(data_test)
    random.shuffle(data_test)
    print('data_test.shape:', data_test.shape)
    data_test_len = data_test.shape[0]
    
    # check data for label errors
    for item in data:
        if item['label'] not in MAPPING_AGGRESSION_MULTI:
            print(item['label'])
        if 'label2' in item:
            if item['label2'] not in MAPPING_AGGRESSION_MULTI:
                print(item['label2'])

    # check class distribution over all corpora
    c = Counter(['-'.join(str(MAPPING_AGGRESSION_MULTI[item['label2']])) if 'label2' in item else '-'.join(str(MAPPING_AGGRESSION_MULTI[item['label']])) for item in data])
    print('class distribution:', c)


    # Lookup embeddings
    #embedding = BPEmb(lang='de', vs=bpe_vs, dim=emb_dim)
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
    #os.system('rm -r logs/*')


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
