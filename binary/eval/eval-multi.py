#!/usr/bin/env python

"""
evaluate multiple models and plot results
"""

import os
import re
import gc
import html
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# tokenization
from bpemb import BPEmb
import emoji

from datetime import datetime
from tqdm import tqdm
from joblib import dump, load
from glob import glob
import json

import matplotlib.pyplot as plt

# fixed seed
#from numpy.random import seed
#seed(1)
#tf.random.set_seed(2)

# parameters
MODEL_PATH = 'en-binary'  # en-binary, de-binary
LANGUAGE = 'en'  # en, de

#data_len = 100000  # randomly sample data_len rows
#threshold = 0.5

emb_dim = 300  # embeddings dimensions
seq_len = 120  # sequence length

BPE_MODEL = '../../bpemb/'+LANGUAGE+'-50k-300d.model'
BPE_EMBEDDING = '../../bpemb/'+LANGUAGE+'-50k-300d-vectors.txt'


def load_minitest():
    if LANGUAGE == 'de':
        DATA_PATH = 'test-de.csv'
    elif LANGUAGE == 'en':
        DATA_PATH = 'test-en.csv'
    else:
        print('unknown language code')
    
    data = germeval(files=[DATA_PATH], local=True)

    return data


def load_madlibs():
    if LANGUAGE == 'de':
        DATA_PATH = 'bias_madlibs_89k-German.csv'
    elif LANGUAGE == 'en':
        DATA_PATH = 'bias_madlibs_89k.csv'
    else:
        print('unknown language code')

    data = debias_madlibs(files=[DATA_PATH], local=True)

    return data


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


# custom metrics, vgl. https://datascience.stackexchange.com/a/45166
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# transformer cf. https://keras.io/examples/nlp/text_classification_with_transformer/

# Implement multi head self attention as a Keras layer
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    
    def get_config(self):
        config = super().get_config() #.copy()
        #config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config


# Implement a Transformer block as a layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(self.embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config() #.copy()
        #config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config


# Implement embedding layer
# Two seperate embedding layers, one for tokens, one for token index (positions).
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
        })
        return config


if __name__ == '__main__':
    
    
    # Load data
    print('* Loading data')
    
    #data = load_madlibs()
    data = load_minitest()

    random.shuffle(data)
    print('data.shape:', data.shape)
    

    # Lookup embeddings
    #embedding = BPEmb(lang=LANGUAGE, vs=bpe_vs, dim=emb_dim)
    embedding = BPEmb(model_file=BPE_MODEL, emb_file=BPE_EMBEDDING, dim=emb_dim)

    print('* Looking up embeddings')
    data_x, data_y = create_embeddings(embedding, data)


    # tmp save / load
    dump(data_x, 'data_x.joblib')
    dump(data_y, 'data_y.joblib')
    
    data_x = load('data_x.joblib')
    data_y = load('data_y.joblib')
    data_len = len(data_x)


    # Pad the x matrix (and possibly convert to 16bit)
    print('* Padding and saving data')
    fp = np.memmap('data_x.dat', dtype='float32', mode='w+', shape=(data_len, seq_len, emb_dim))
    
    # avoid oom: add padded sequences in batches
    batch_len = 2000
    num_batches = int(len(data_x) / batch_len)
    for i in tqdm(range(num_batches)):
        write_start = batch_len * i
        write_end = batch_len * (i + 1)
        fp[write_start:write_end] = pad_x_matrix(data_x[:batch_len], max_tokens=seq_len)
        data_x = data_x[batch_len:]
        #gc.collect()
    if len(data_x) > 0:
        fp[batch_len * num_batches:] = pad_x_matrix(data_x, max_tokens=seq_len)  # add rest

    data_y = np.array(data_y)
    np.save('data_y.npy', data_y)
    


    # Load preprocessed data
    print('* Loading data')
    fp = np.memmap('data_x.dat', dtype='float32', mode='r', shape=(data_len, seq_len, emb_dim))
    data_y = np.load('data_y.npy')
    print('data.shape:', fp.shape)
    


    # evaluate
    models_scores = {}
    metrics_names = []
    
    for path in glob(MODEL_PATH+'/*.h5'):

        # load model as is
        #model = keras.models.load_model(path,
        #                                custom_objects={'f1_m': f1_m,
        #                                                'precision_m': precision_m,
        #                                                'recall_m': recall_m,
        #                                                'TransformerBlock': TransformerBlock,
        #                                                'MultiHeadSelfAttention': MultiHeadSelfAttention})

        # add new metrics to model
        model = keras.models.load_model(path,
                                        compile=False,
                                        custom_objects={'f1_m': f1_m,
                                                        'precision_m': precision_m,
                                                        'recall_m': recall_m,
                                                        'TransformerBlock': TransformerBlock,
                                                        'MultiHeadSelfAttention': MultiHeadSelfAttention})
        model.compile(loss='binary_crossentropy',
                        optimizer=keras.optimizers.RMSprop(lr=0.001, centered=True),
                        #metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy_0.99', threshold=0.99),
                        #        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy_0.9', threshold=0.9),
                        #        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy_0.5', threshold=0.5),
                        metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy_0.5', threshold=0.5),
                        tf.keras.metrics.AUC(),
                        f1_m,
                        precision_m,
                        recall_m])
        
        print('\nModel evaluation:', path)
        start_eval = datetime.now()
        
        scores = model.evaluate(fp, data_y, verbose=0)
        
        runtime_eval = datetime.now()-start_eval
        for i, score in enumerate(scores):
            print("%s: %.2f%%" % (model.metrics_names[i], scores[i]*100))
        print('\nRuntime: ', runtime_eval.total_seconds() / 60, ' minutes')
        

        # save for plotting
        metrics_names = model.metrics_names
        models_scores[path] = scores



    dump(models_scores, 'models_scores.joblib')
    #dump(metrics_names, 'metrics_names.joblib')
    

    models_scores = load('models_scores.joblib')
    #metrics_names = load('metrics_names.joblib')
    metrics_names = ['loss', '0.5', 'auc', 'f1', 'prec', 'rec']  #'0.99', '0.9'



    # plot
    print(len(models_scores), 'models tested')
    plot_len = 5  #int(len(models_scores) / 3)
    print('arranging in', plot_len, 'x', plot_len, 'grid')

    fig, axs = plt.subplots(plot_len, plot_len, figsize=(40, 40))
    
    j = 0
    for i, (model_id, model_scores) in enumerate(models_scores.items()):
        if i >= plot_len and i < plot_len*2:
            i -= plot_len
        elif i >= plot_len*2 and i < plot_len*3:
            i -= plot_len*2
        elif i >= plot_len*3 and i < plot_len*4:
            i -= plot_len*3
        elif i >= plot_len*4 and i < plot_len*5:
            i -= plot_len*4
        #print(i, j)

        axs[i, j].bar(metrics_names, model_scores)
        #axs[i, j].plot(metrics_names, model_scores)
        title = model_id.replace('saved_models\model-', '').replace('.h5', '')
        axs[i, j].set_title(title)

        if i == plot_len-1:
            j += 1  # if the row is finished, move to the next
    

    plt.savefig('eval-multi.png')

    # bell sound
    print('\a')




    """
    plt.figure()
    for model_id, model_scores in models_scores.items():
        plt.subplot(211)
        plt.bar(metrics_names, model_scores)
        plt.title = model_id
        #plt.show()
        plt.savefig('eval-multi.png')
    """

    # predict (obsolet -> bei evaluation: threshold für binary_accuracy)
    """
        model = keras.models.load_model(path, custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m, 'TransformerBlock': TransformerBlock, 'MultiHeadSelfAttention': MultiHeadSelfAttention})
        results_statistic = {'correct': 0, 'incorrect': 0}
        
        print('\nModel evaluation:', path)
        start_eval = datetime.now()

        for x, y in zip(fp, data_y):
            x = np.reshape(x, (1,) + x.shape)
            score = model.predict(x)

            if score[0][0] > threshold and y == 1:
                results_statistic['correct'] +=1
            elif score[0][0] < threshold and y == 0:
                results_statistic['correct'] +=1
            else:
                results_statistic['incorrect'] +=1

        runtime_eval = datetime.now()-start_eval
        total = results_statistic['correct'] + results_statistic['incorrect']
        correct_percent = results_statistic['correct'] / (total / 100)
        incorrect_percent = results_statistic['incorrect'] / (total / 100)
        print('Accuracy with threshold', threshold, ':', results_statistic, '= {0:.2f}'.format(correct_percent), '% : {0:.2f}'.format(incorrect_percent), '%')
        print('\nRuntime: ', runtime_eval.total_seconds() / 60, ' minutes')
    """
