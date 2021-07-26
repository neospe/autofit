#!/usr/bin/env python

"""
evaluate a model using a few manually specified examples
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
import json

from dataload.loader import *
from dataload.preproc import *

# fixed seed
#from numpy.random import seed
#seed(1)
#tf.random.set_seed(2)

# parameters
LANGUAGE = 'en'
MODEL_PATH = 'en-multi3/multi_model-2021-04-29-17-37-474571183-best-val-f1.h5'

INTERACTIVE = True

no_of_classes = 5
seq_len = 120  # sequence length
emb_dim = 300  # embeddings dimensions

BPE_MODEL = '../../bpemb/'+LANGUAGE+'-50k-300d.model'
BPE_EMBEDDING = '../../bpemb/'+LANGUAGE+'-50k-300d-vectors.txt'


def create_embeddings(embedding, text):
    array_of_vecs = embedding.embed(text)        
    list_of_vecs = np.split(array_of_vecs, array_of_vecs.shape[0], axis=0)
    list_of_vecs = [v.flatten() for v in list_of_vecs]

    return [list_of_vecs]


def pad_x_matrix(x_matrix, max_tokens=seq_len):
    for sentenceIdx in range(len(x_matrix)):
        sent = x_matrix[sentenceIdx]
        sentence_vec = np.array(sent, dtype=np.float32)  # np.float16

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
    
    model = keras.models.load_model(MODEL_PATH, custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m, 'TransformerBlock': TransformerBlock, 'MultiHeadSelfAttention': MultiHeadSelfAttention})
    #embedding = BPEmb(lang=LANGUAGE, vs=bpe_vs, dim=emb_dim)
    embedding = BPEmb(model_file=BPE_MODEL, emb_file=BPE_EMBEDDING, dim=emb_dim)
    
    if INTERACTIVE == True:
        while True:
            print('enter text:')
            text = input()
            if text != '':
                text = clean_str(text)
                x = create_embeddings(embedding, text)
                x = pad_x_matrix(x, max_tokens=seq_len)
                score = model.predict(x)
                for sc in score:
                    for s in sc:
                        print('{:.20f}'.format(s))

    else:
        if LANGUAGE == 'de':
            DATA_PATH = 'test-de.csv'
        elif LANGUAGE == 'en':
            DATA_PATH = 'test-en.csv'
        else:
            print('unknown language code')

        data = germeval(files=[DATA_PATH], local=True)
        for post in data:
            label_str = post['label2']
            label = [0 for i in range(no_of_classes)]
            label[MAPPING_AGGRESSION_MULTI[label_str]] = 1
            text = clean_str(post['text'])
            x = create_embeddings(embedding, text)
            x = pad_x_matrix(x, max_tokens=seq_len)
            #print(x.shape)
            print(text, ' -- ', post['label'])
            score = model.predict(x)
            for sc in score:
                for s in sc:
                    print('{:.20f}'.format(s))
