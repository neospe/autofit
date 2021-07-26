#!/usr/bin/env python

import os
import re
import gc
import html
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

# training
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# automation
from sacred import Experiment, SETTINGS
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds  # make stdout logs smaller
from ray.tune.integration.keras import TuneReportCallback
from ray import tune


# INIT + PARAMETERS

from run_en import no_of_classes, emb_dim, seq_len, validation_split, EXPERIMENT_NAME

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver(url='127.0.0.1:27017', db_name='sacred'))
ex.capture_out_filter = apply_backspaces_and_linefeeds
SETTINGS.CAPTURE_MODE = 'no'


# TRAINING

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
    f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1


class SacredMetrics(keras.callbacks.Callback):

    def __init__(self, run):
        super(SacredMetrics, self).__init__()
        self.run = run
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.run.log_scalar('val_loss', logs.get('val_loss'))
        self.run.log_scalar('val_acc', logs.get('val_accuracy'))
        self.run.log_scalar('val_auc', logs.get('val_auc'))
        self.run.log_scalar('val_f1', logs.get('val_f1_m'))
        self.run.log_scalar('val_prec', logs.get('val_precision_m'))
        self.run.log_scalar('val_rec', logs.get('val_recall_m'))

    def on_train_end(self, logs=None):
        pass


class TestMetrics(keras.callbacks.Callback):
    """
    evaluate model with test data every epoch, report to sacred + tune, and checkpoint model
    """

    def __init__(self, run, fp_test, data_y_test, outfile):
        super(TestMetrics, self).__init__()
        self.run = run
        self.fp_test = fp_test
        self.data_y_test = data_y_test
        self.outfile = outfile
        self.test_f1 = []
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):

        scores = self.model.evaluate(self.fp_test, self.data_y_test, verbose=0)

        # report to sacred
        self.run.log_scalar('test_loss', scores[0])
        self.run.log_scalar('test_acc', scores[1])
        self.run.log_scalar('test_auc', scores[2])
        self.run.log_scalar('test_f1', scores[3])
        self.run.log_scalar('test_prec', scores[4])
        self.run.log_scalar('test_rec', scores[5])

        test_f1 = scores[3]
        tune.report(test_f1=test_f1)  # report to tune

        # model checkpoint
        if all(test_f1 > m for m in self.test_f1):  # if largest test_f1 score
            print('test_f1 improved, saving model to', self.outfile)
            self.model.save(self.outfile)
        else:
            print('test_f1 did not improve')
        self.test_f1.append(test_f1)

        # stop training if test_loss exceeds threshold
        if scores[0] >= 1.0:
            print('test_loss exceeding threshold. stopping early.')
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        pass


# transformer cf. https://keras.io/examples/nlp/text_classification_with_transformer/
class MultiHeadSelfAttention(layers.Layer):
    # Implement multi head self attention as a Keras layer
    
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


class TransformerBlock(layers.Layer):
    # Implement a Transformer block as a layer

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


class TokenAndPositionEmbedding(layers.Layer):
    # Implement embedding layer
    # Two seperate embedding layers, one for tokens, one for token index (positions).
    
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


@ex.main
def train_model(_run, _config):

    if _config['custom_seed'] != False:
        np.random.seed(_config['custom_seed'])
        tf.random.set_seed(_config['custom_seed'])

    assert _config['dense_0_size'] > _config['dense_1_size'] > _config['dense_2_size'] > _config['dense_3_size']

    # Load data
    print('* Loading data')
    data_len = load('data_len.joblib')
    fp = np.memmap('data_x.dat', dtype='float32', mode='r', shape=(data_len, seq_len, emb_dim))
    data_y = np.load('data_y.npy')
    print('data.shape:', fp.shape)

    # load extra test data
    data_test_len = load('data_test_len.joblib')
    fp_test = np.memmap('data_x_test.dat', dtype='float32', mode='r', shape=(data_test_len, seq_len, emb_dim))
    data_y_test = np.load('data_y_test.npy')
    print('data_test.shape:', fp_test.shape)


    # prepare for training
    print('* Training')
    start = datetime.now()
    run_id = 'multi_model-'+datetime.today().strftime('%Y-%m-%d-%H-%M')+'-'+str(_config['seed'])
    

    # Architektur

    transformer_block = TransformerBlock(emb_dim, _config['transformer_num_heads'], _config['transformer_ff_dim'], rate=_config['transformer_dropout'])

    _input = layers.Input(shape=(seq_len, emb_dim))

    if _config['dropout_1_on'] == True:
        x = layers.Dropout(_config['dropout_1'])(_input)
        if _config['transformer_on'] == True:
            x = transformer_block(x)
        else:
            x = layers.Dense(_config['dense_0_size'], activation='relu')(x)  #400
    else:
        if _config['transformer_on'] == True:
            x = transformer_block(_input)
        else:
            x = layers.Dense(_config['dense_0_size'], activation='relu')(_input)  #400
    
    if _config['pooling_on'] == True:
        x = layers.GlobalAveragePooling1D()(x)
    
    if _config['dense_1_on'] == True:
        x = layers.Dense(_config['dense_1_size'], activation='relu')(x)
    
    if _config['dropout_2_on'] == True:
        x = layers.Dropout(_config['dropout_2'])(x)
    
    if _config['dense_2_on'] == True:
        x = layers.Dense(_config['dense_2_size'], activation='relu')(x)
    
    if _config['dropout_3_on'] == True:
        x = layers.Dropout(_config['dropout_3'])(x)
    
    if _config['dense_3_on'] == True:
        x = layers.Dense(_config['dense_3_size'], activation='relu')(x)
    
    if _config['batch_norm_on'] == True:
        x = layers.BatchNormalization()(x)

    #_output = layers.Dense(1, activation='sigmoid')(x)  # binary classification
    _output = layers.Dense(no_of_classes, activation='softmax')(x)  # multi-class classification

    model = keras.Model(_input, _output)


    # Callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir='logs/'+run_id)
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=_config['early_stopping_patience'], verbose=1, mode='min')

    outfile = 'saved_models/'+run_id+'-best-val-f1.h5'
    model_checkpoint1 = keras.callbacks.ModelCheckpoint(outfile, monitor='val_f1_m', verbose=1, save_best_only=True, mode='max')

    sacred_metrics = SacredMetrics(_run)
    
    tune_metrics = TuneReportCallback()

    outfile2 = 'saved_models/'+run_id+'-best-test-f1.h5'
    
    # extra test data -> siehe auch unten in model.fit()
    test_metrics = TestMetrics(_run, fp_test, data_y_test, outfile2)


    # Optimizer
    
    if _config['optimizer'] == 'nadam':
        optimizer = keras.optimizers.Nadam(lr=_config['learning_rate'])  #, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.0)  #0.0005, 0.0001, defaults: lr=0.002, schedule_decay=0.004
    elif _config['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(lr=_config['learning_rate'])  #0.0005
    elif _config['optimizer'] == 'sgd':
        optimizer = keras.optimizers.SGD(lr=_config['learning_rate'])  #0.001
    elif _config['optimizer'] == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(lr=_config['learning_rate'], centered=True)  #0.0005


    # Train model

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.AUC(), f1_m, precision_m, recall_m])  # binary_crossentropy, poisson, tf.keras.losses.kullback_leibler_divergence

    model.fit(fp, data_y, epochs=_config['epochs'], batch_size=_config['batch_size'], validation_split=validation_split, callbacks=[tensorboard, model_checkpoint1, sacred_metrics, tune_metrics, test_metrics, early_stopping])  # ANM: early_stopping muss das letzte objekt der liste sein

    model.summary()

    print('\n* Model saved to:')
    print(outfile)


    # add saved models as artefact to sacred
    _run.add_artifact(outfile)
    _run.add_artifact(outfile2)


    # Evaluate model -> siehe TestMetrics class
    """
    print('\n* Model evaluation:')
    model_eval = keras.models.load_model(outfile, custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m, 'TransformerBlock': TransformerBlock, 'MultiHeadSelfAttention': MultiHeadSelfAttention})
    start_eval = datetime.now()
    scores = model_eval.evaluate(fp_test, data_y_test, verbose=0)
    runtime_eval = datetime.now()-start_eval
    for i, score in enumerate(scores):
        print("%s: %.2f%%" % (model_eval.metrics_names[i], scores[i]*100))
    print('\nRuntime: ', runtime_eval.total_seconds() / 60, ' minutes')
    print('\n\n')


    # add metrics to sacred
    _run.log_scalar('test_loss', scores[0])
    _run.log_scalar('test_acc', scores[1])
    _run.log_scalar('test_auc', scores[2])
    _run.log_scalar('test_f1', scores[3])
    _run.log_scalar('test_prec', scores[4])
    _run.log_scalar('test_rec', scores[5])

    # add as artefact to sacred
    _run.add_artifact(outfile)


    tune.report(test_f1=scores[3])  # training_iteration=epoch
    """
