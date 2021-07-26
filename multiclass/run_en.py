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

from config_en import *

# system
os.system('$env:CUDA_VISIBLE_DEVICES="0,1"')
os.system('$env:TUNE_DISABLE_STRICT_METRIC_CHECKING="1"')
DB_ADDR = '127.0.0.1:27017'
DB_NAME = 'datalake'
LOG_WORKERS = False
ray.init(num_cpus=8, num_gpus=2, ignore_reinit_error=True, log_to_driver=LOG_WORKERS)

# experiment
EXPERIMENT_NAME = 'EN-MULTI-default_4'
CONFIG = default_4  # siehe config.py
NUM_SAMPLES = 10  #10  # how many configurations to sample = how many trials to run

# TODO: zwecks versionierung in config dict definieren?
no_of_classes = 5
emb_dim = 300
seq_len = 120
validation_split = 0.2

#==================

# PATHS

BPE_MODEL = '../bpemb/en-50k-300d.model'
BPE_EMBEDDING = '../bpemb/en-50k-300d-vectors.txt'

DATA_1 = 'hate-speech-and-offensive-language/data/labeled_data.csv'  # 24783 items

DATA_2_ANN = 'hatespeech/NAACL_SRW_2016.csv'
DATA_2_TXT = 'AutomaticDetectionofHateSpeech/hatespeech.csv'  # 16035 items

DATA_3_ANN = 'hate-speech-dataset/annotations_metadata.csv'
DATA_3_TXT = 'hate-speech-dataset/all_files'  # 10944 items

DATA_4_TXT = 'OffensEval-2019/offenseval-training-v1.tsv'  # 13240 items
DATA_4_CORR = 're-labelings/OffensEval-2019-profanity-predicted-edit.csv'

DATA_5_TXT = 'founta-corpus/hatespeechtwitter-tweets.csv'  # 80.000 items
DATA_5_ANN = 'founta-corpus/hatespeechtwitter.tsv'
DATA_5_CORR = 're-labelings/founta-corpus-profanity-predicted-edit.csv'

#DATA_6_TXT = 'delft/textClassification/toxic/train.csv'  # 159.571 items
#DATA_6_1_TXT = 'delft/textClassification/toxic/test.csv'
#DATA_6_1_ANN = 'delft/textClassification/toxic/test_labels.csv'  # 153.163 items

DATA_7 = 'HateXplain/Data/dataset.json'

DATA_8_1_TXT = 'SOLID/semeval_test/test_a_tweets.tsv'
DATA_8_1_ANN = 'SOLID/semeval_test/test_a_labels.csv'
DATA_8_2_TXT = 'SOLID/extended_test/test_a_tweets_all.tsv'
DATA_8_2_ANN = 'SOLID/extended_test/test_a_labels_all.csv'
DATA_8_CORR = 're-labelings/SOLID-testdata-profanity-predicted-edit.csv'

DATA_9_TXT = 'OLIDv1.0/testset-levela.tsv'
DATA_9_ANN = 'OLIDv1.0/labels-levela.csv'

DATA_10 = 'MalignantCommentClassification/train.csv'

DEBIAS_MADLIBS = 'social-bias/selection/bias_madlibs_89k.csv'
DATA_THREAT = 'hexis-english-threats/hexis-english-threats.txt'
SPAM_1 = 'smsspamcollection/SMSSpamCollection.txt'
DATA_MINITEST = 'eval/test-en.csv'  # loader: local=True

AUG_BACKLOG_PATH = ['backlog.json']
AUGS_TO_USE = [DATA_1, DATA_2_ANN, DATA_3_ANN, DATA_4_TXT, DATA_5_TXT, DATA_7, DATA_8_1_TXT]  #DATA_9_TXT

# ANM: extra testdaten müssen in einem eigenen directory liegen, sonst wird es mit den augmentations zusammen geladen (geht nach dataset/directory name, e.g. 'GermEval-2019-Data') -> evtl. TODO: beim laden der augmentations nochmal mit pfad der testdaten abgleichen u. diese excluden


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
    data = []
    data += hate1(files=[DATA_1])
    data += hate2(files=[DATA_2_TXT, DATA_2_ANN])
    data += hate3(files=[DATA_3_ANN, DATA_3_TXT])
    data += hate4(files=[DATA_5_TXT, DATA_5_ANN, DATA_5_CORR])
    data += offense19(files=[DATA_4_TXT, DATA_4_CORR])
    data += offense20(files=[DATA_8_1_TXT, DATA_8_1_ANN, DATA_8_2_TXT, DATA_8_2_ANN, DATA_8_CORR])
    data += hatexplain(files=[DATA_7])
    data += malignant(files=[DATA_10])
    data += spam1(files=[SPAM_1])
    data += debias_madlibs(files=[DEBIAS_MADLIBS])
    data += germeval(files=[DATA_THREAT])
    data += augmentations(files=AUG_BACKLOG_PATH+AUGS_TO_USE)
    data = np.asarray(data)
    random.shuffle(data)
    print('data.shape:', data.shape)
    data_len = data.shape[0]

    # extra test data
    data_test = olid(files=[DATA_9_TXT, DATA_9_ANN])
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
    #embedding = BPEmb(lang='en', vs=bpe_vs, dim=emb_dim)
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
    
    from nn_en import ex
    
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
    #search_alg = HyperOptSearch(metric='loss', mode='min')
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
