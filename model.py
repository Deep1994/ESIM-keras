# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:15:58 2019

@author: Deep
"""

import numpy as np
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K
from layers import *

# see https://www.kaggle.com/lamdang/dl-models
def esim(pretrained_embedding='./glove.npy', 
         maxlen=25, 
         max_char_len=15,
         lstm_dim=128, 
         dense_dim=300, 
         dense_dropout=0.5,
         charsize=None):
             
    # Based on arXiv:1609.06038
    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))
    
    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding, mask_zero=False)
    bn = BatchNormalization(axis=2)
    q1_embed = bn(embedding(q1))
    q2_embed = bn(embedding(q2))
    
    # char embedding
    q1_char = Input(name='q1_char', shape=(maxlen, max_char_len,))
    q2_char = Input(name='q2_char', shape=(maxlen, max_char_len,))
    
    char_embedding = create_char_embedding(charsize=1637, maxlen=maxlen, max_char_len=max_char_len, char_embedding_dim=100)
    char_bn = BatchNormalization(axis=3)
    q1_char_embed = char_bn(char_embedding(q1_char))
    q2_char_embed = char_bn(char_embedding(q2_char))
    
    char_lstm = Bidirectional(LSTM(50))
    q1_char_embed = TimeDistributed(char_lstm)(q1_char_embed)
    q2_char_embed = TimeDistributed(char_lstm)(q2_char_embed)    
    

    # word concat char
    q1_embed = Concatenate(axis=2)([q1_embed, q1_char_embed])
    q2_embed = Concatenate(axis=2)([q2_embed, q2_char_embed])

    # Encode
    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    
    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)]) 
       
    compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)
    
    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    
    dense = BatchNormalization()(merged)
    dense = Dense(256, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(32, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)
    
#    model = Model(inputs=[q1, q2], outputs=out_)
    model = Model(inputs=[q1, q2, q1_char, q2_char], outputs=out_)

    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model




