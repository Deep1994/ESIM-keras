# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:23:45 2019

@author: Deep
"""
from nltk.tokenize import TweetTokenizer
import collections
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from preprocessing import clean_text

def data_reader(data_file):
    q1 = []
    q2 = []
    labels = []
    with open(data_file, "r", errors="ignore") as f:
#        next(f)
        for line in f:
            line = line.strip().split("\t")
            q1.append(clean_text(line[1]))
            q2.append(clean_text(line[2]))
            labels.append(int(line[0]))
    return q1, q2, labels


def get_vocab(sentences_list):
    
    tt = TweetTokenizer()
    tokenized_sentences = []
    
    for sent in sentences_list:
         tokenized_sentences.append(tt.tokenize(sent))
    
    wordcounts = collections.Counter()
    for tokenized_sent in tokenized_sentences:
        for word in tokenized_sent:
            wordcounts[word] += 1
            
    words = [wordcount[0] for wordcount in wordcounts.most_common()]
    word2idx = {w: i+3 for i, w in enumerate(words)}
    word2idx["_pad_"] = 0
    word2idx["_mask_"] = 1
    word2idx["_unk_"] = 2
    
    idx2word  = dict((k,v) for v,k in word2idx.items())
    
    return word2idx, idx2word


def get_vectorized_sentences(sentences_list, word2idx, max_sent_len=25, is_test=False):
    
    tt = TweetTokenizer()
    tokenized_sentences = []
    vectorized_sentences = []
    
    for sent in sentences_list:
         tokenized_sentences.append(tt.tokenize(sent))
    
    if not is_test:
        for tokenized_sent in tokenized_sentences:
            # truncate the sentence to the max_sent_len 
            if len(tokenized_sent) > max_sent_len:
                tokenized_sent = tokenized_sent[:max_sent_len] 
            vectorized_sentences.append([word2idx[word] for word in tokenized_sent])
    else:
        for tokenized_sent in tokenized_sentences:
            # truncate the sentence to the max_sent_len 
            if len(tokenized_sent) > max_sent_len:
                tokenized_sent = tokenized_sent[:max_sent_len] 
            ts = []
            for word in tokenized_sent:
                if word in word2idx:
                    ts.append(word2idx[word])
                else:
                    ts.append(word2idx["_unk_"])
            
            vectorized_sentences.append(ts)
    # padding
    vectorized_sentences = pad_sequences(vectorized_sentences, maxlen=max_sent_len, padding="post")
            
    return vectorized_sentences

def get_char(sentences_list):
    
    tt = TweetTokenizer()
    tokenized_sentences = []
    
    for sent in sentences_list:
         tokenized_sentences.append(tt.tokenize(sent))
    
    char_per_word = []
    char_words = []
    char_sents = []
    
    for tokenized_sent in tokenized_sentences:
        for word in tokenized_sent:
            for char in word:
                char_per_word.append(char)
                
#            if len(char_per_word) > max_char_len:
#                char_per_word = char_per_word[:max_char_len]
                    
            char_words.append(char_per_word)
            char_per_word = []
        
        char_sents.append(char_words)
        char_words = []
     
    charcounts = collections.Counter()
    for sent in char_sents:
        for word in sent:
            for char in word:
                charcounts[char] += 1
                
    chars = [charcount[0] for charcount in charcounts.most_common()]
    char2idx = {c: i+2 for i, c in enumerate(chars)}  
    char2idx["_pad_"] = 0 
    char2idx["_unk_"] = 1
      
    idx2char  = dict((k,v) for v,k in char2idx.items())
    
    return char2idx, idx2char


def get_vectorized_char_sentences(sentences_list, char2idx, max_sent_len=25, max_char_len=15, is_test=False):
    
    tt = TweetTokenizer()
    tokenized_sentences = []
    vectorized_char_sentences = []
    
    for sent in sentences_list:
         tokenized_sentences.append(tt.tokenize(sent))
    
    char_per_word = []
    char_words = []
    char_sents = []
    char_all_sents = []
    
    if not is_test:
        for tokenized_sent in tokenized_sentences:
            # truncate the sentence to the max_sent_len 
            if len(tokenized_sent) > max_sent_len:
                tokenized_sent = tokenized_sent[:max_sent_len]
            for word in tokenized_sent:
                for char in word:
                    char_per_word.append([char2idx[char]])
                    if len(char_per_word) > max_char_len:
                        char_per_word = char_per_word[:max_char_len]
                
                char_words.append(char_per_word)
                char_per_word = []
        
            char_sents.append(char_words)
            char_words = []
            
        char_per_word = []  
        char_per_sent = [] 
        for s in char_sents:
            for w in s:
                for c in w:
                    for e in c:
                        char_per_word.append(e)
                char_per_sent.append(char_per_word)
                char_per_word = []
            char_all_sents.append(char_per_sent)
            char_per_sent = [] 
    
    else:
        for tokenized_sent in tokenized_sentences:
            # truncate the sentence to the max_sent_len 
            if len(tokenized_sent) > max_sent_len:
                tokenized_sent = tokenized_sent[:max_sent_len]
            for word in tokenized_sent:
                for char in word:
                    if char in char2idx:
                        char_per_word.append(char2idx[char])
                    else:
                        char_per_word.append(char2idx["_unk_"])
                        
                    if len(char_per_word) > max_char_len:
                        char_per_word = char_per_word[:max_char_len]
                char_words.append(char_per_word)
                char_per_word = []
                
            char_all_sents.append(char_words)
            char_words = []
                            
    # paddding
    for sent in char_all_sents:
        while len(sent) < max_sent_len:
#            sent.insert(0, []) # 在句首插入
            sent.append([]) # 在句末插入
        pad_char_sent = pad_sequences(sent, maxlen=max_char_len, padding="post")
        vectorized_char_sentences.append(pad_char_sent)
    
    return vectorized_char_sentences

def load_glove(file):
    """Loads GloVe vectors in numpy array.
    Args:
        file (str): a path to a glove file.
    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with open(file, encoding="utf8", errors='ignore') as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector

    return model

def filter_embeddings(embeddings, vocab, dim):
    """Loads word vectors in numpy array.
    Args:
        embeddings (dict): a dictionary of numpy array.
        vocab (dict): word_index lookup table.
    Returns:
        numpy array: an array of word embeddings.
    """
    if not isinstance(embeddings, dict):
        return
    _embeddings = np.zeros([len(vocab), dim])
    for word in vocab:
        word_idx = vocab[word]
        if word in embeddings:
            _embeddings[word_idx] = embeddings[word]
        else:
            _embeddings[word_idx] = np.random.uniform(-1, 1, dim)

    return _embeddings

