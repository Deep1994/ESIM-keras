# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:26:19 2019

@author: Deep
"""
from preprocessing import clean_text
from utils import *
import numpy as np
from model import esim
from keras.callbacks import ModelCheckpoint, EarlyStopping

def main():
    train_file = "./dataset/train.tsv"
    dev_file = "./dataset/dev.tsv"
    test_file = "./dataset/test.tsv"
    glove_file = "./glove.840B.300d.txt"
    embed_dim = 300
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    VERBOSE = 1
    PATIENCE = 5
    
    train_q1, train_q2, train_labels = data_reader(train_file)
    dev_q1, dev_q2, dev_labels = data_reader(dev_file)
    test_q1, test_q2, test_labels = data_reader(test_file)
   
    sentences_list = train_q1 + train_q2 + dev_q1 + dev_q2
    word2idx, idx2word = get_vocab(sentences_list)
    # char
    char2idx, idx2char = get_char(sentences_list)
    
    train_vectorized_q1 = get_vectorized_sentences(train_q1, word2idx)
    train_vectorized_q2 = get_vectorized_sentences(train_q2, word2idx)
    train_labels = np.array(train_labels)
    
    dev_vectorized_q1 = get_vectorized_sentences(dev_q1, word2idx)
    dev_vectorized_q2 = get_vectorized_sentences(dev_q2, word2idx)
    dev_labels = np.array(dev_labels)
    
    test_vectorized_q1 = get_vectorized_sentences(test_q1, word2idx, is_test=True)
    test_vectorized_q2 = get_vectorized_sentences(test_q2, word2idx, is_test=True)
    test_labels = np.array(test_labels)
    
#    np.save("./train_vectorized_q1.npy", train_vectorized_q1)
#    np.save("./train_vectorized_q2.npy", train_vectorized_q2)
#    np.save("./train_labels.npy", train_labels)
    
#    np.save("./dev_vectorized_q1.npy", dev_vectorized_q1)
#    np.save("./dev_vectorized_q2.npy", dev_vectorized_q2)
#    np.save("./dev_labels.npy", dev_labels)

#    np.save("./test_vectorized_q1.npy", test_vectorized_q1)
#    np.save("./test_vectorized_q2.npy", test_vectorized_q2)
#    np.save("./test_labels.npy", test_labels)

#    train_vectorized_q1 = np.load("./train_vectorized_q1.npy")
#    train_vectorized_q2 = np.load("./train_vectorized_q2.npy")
#    train_labels = np.load("./train_labels.npy")
#    
#    dev_vectorized_q1 = np.load("./dev_vectorized_q1.npy")
#    dev_vectorized_q2 = np.load("./dev_vectorized_q2.npy")
#    dev_labels = np.load("./dev_labels.npy")
#    
#    test_vectorized_q1 = np.load("./test_vectorized_q1.npy")
#    test_vectorized_q2 = np.load("./test_vectorized_q2.npy")
#    test_labels = np.load("./test_labels.npy")
    
    # char
    train_vectorized_char_q1 = get_vectorized_char_sentences(train_q1, char2idx)
    train_vectorized_char_q2 = get_vectorized_char_sentences(train_q2, char2idx)
    
    dev_vectorized_char_q1 = get_vectorized_char_sentences(dev_q1, char2idx)
    dev_vectorized_char_q2 = get_vectorized_char_sentences(dev_q2, char2idx)
    
    test_vectorized_char_q1 = get_vectorized_char_sentences(test_q1, char2idx, is_test=True)
    test_vectorized_char_q2 = get_vectorized_char_sentences(test_q2, char2idx, is_test=True)
    
#    np.save("./train_vectorized_char_q1.npy", train_vectorized_char_q1)
#    np.save("./train_vectorized_char_q2.npy", train_vectorized_char_q2)
#    
#    np.save("./dev_vectorized_char_q1.npy", dev_vectorized_char_q1)
#    np.save("./dev_vectorized_char_q2.npy", dev_vectorized_char_q2)
#
#    np.save("./test_vectorized_char_q1.npy", test_vectorized_char_q1)
#    np.save("./test_vectorized_char_q2.npy", test_vectorized_char_q2)
    
#    train_vectorized_char_q1 = np.load("./train_vectorized_char_q1.npy")
#    train_vectorized_char_q2 = np.load("./train_vectorized_char_q2.npy")
#    
#    dev_vectorized_char_q1 = np.load("./dev_vectorized_char_q1.npy")
#    dev_vectorized_char_q2 = np.load("./dev_vectorized_char_q2.npy")
#    
#    test_vectorized_char_q1 = np.load("./test_vectorized_char_q1.npy")
#    test_vectorized_char_q2 = np.load("./test_vectorized_char_q2.npy")
    
    # load glove model
    glove_model = load_glove(glove_file)
    embeddings = filter_embeddings(glove_model, word2idx, embed_dim)
#    np.save("./glove.npy", embeddings)
#    embeddings = np.load("./glove.npy")
    
    model = esim()
    model.summary()
    
    # set checkpoint
    filepath = './saved_models/weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=VERBOSE,
                                 save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=VERBOSE)
    
    print("*"*20 + 'Training' + "*"*20)
#    history = model.fit([train_vectorized_q1, train_vectorized_q2], train_labels, \
#                        batch_size=BATCH_SIZE, validation_data=([dev_vectorized_q1, dev_vectorized_q2], dev_labels), \
#                        epochs=NUM_EPOCHS, callbacks=[checkpoint, early_stopping])
    
    history = model.fit([train_vectorized_q1, train_vectorized_q2, train_vectorized_char_q1, train_vectorized_char_q2], train_labels, \
                    batch_size=BATCH_SIZE, validation_data=([dev_vectorized_q1, dev_vectorized_q2, dev_vectorized_char_q1, dev_vectorized_char_q2], dev_labels), \
                    epochs=NUM_EPOCHS, callbacks=[checkpoint, early_stopping])
    
#    model.load_weights("./saved_models/weights-improvement-05-0.87600.hdf5")
    
    print("*"*20 + 'Evaluating' + "*"*20)
    score = model.evaluate([test_vectorized_q1, test_vectorized_q2, test_vectorized_char_q1, test_vectorized_char_q2], test_labels, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    
if __name__ == "__main__":
    main()
    
    
    