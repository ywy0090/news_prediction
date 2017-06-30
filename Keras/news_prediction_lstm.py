# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:11:21 2017

@author: wenyu6
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:35:00 2017

@author: wenyu6
"""
import numpy
from gensim.models import word2vec
import re
import jieba
import numpy 
import collections
import os
from io import open
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.cross_validation import train_test_split  
from keras.models import Sequential
from keras.layers import Dropout
import pandas

BASE_DIR = 'D:\\python_project\\news_prediction\\'
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 1000000
EMBEDDING_DIM = 400
VALIDATION_SPLIT = 0.2
TRAIN_EPOCHS = 100


def  getTitleVecs(title):
    vecs=[]
    vec_len=50
    dimension_len=400
    for word in title:
        try:
            vecs.append(model[word].reshape((1,dimension_len)))
        except KeyError:
            continue
    if len(vecs)<vec_len:
        for i in range(vec_len-len(vecs)):
            vecs.append(numpy.zeros((1,dimension_len)))
    elif len(vecs)>vec_len:
        vecs= vecs[0:vec_len]
    vecs = numpy.concatenate(vecs)
    vecs = vecs.reshape(vec_len,dimension_len)
    return numpy.array(vecs, dtype='float')
    

def getWordsIndex(words, words_index):
    sequence=numpy.zeros(MAX_SEQUENCE_LENGTH)
    for i,w in enumerate(words):
        if i >= MAX_SEQUENCE_LENGTH:
            continue
        index = words_index.get(w)
        if index is not None:
        # words not found in embedding index will be all-zeros.
            sequence[i] = index
    return sequence

    
def train_cnn(num_words,embedding_matrix):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    filters=250
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters, 5, activation='relu',input_shape=(50,400)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters, 5, activation='relu'))
    model.add(MaxPooling1D(19))
    model.add(Flatten())
    model.add(Dense(filters, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    loss=list()
    for i in range(TRAIN_EPOCHS):
        hist=model.fit(x_train, y_train,
              batch_size=128,
              epochs=1,
              validation_data=(x_test, y_test))
        loss.append(hist.history['loss'][0])
        json_string = model.to_json()  
        open('cnn_model_architecture.json','w').write(json_string)  
        model.save_weights('cnn_model_weights.h5')
    results = pandas.DataFrame()
    results['cnn']=loss
    results.to_csv('cnn_loss.csv')
    

def train_lstm(num_words,embedding_matrix):
    model = Sequential()
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(50,400)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    loss=list()
    for i in range(TRAIN_EPOCHS):
        hist=model.fit(x_train, y_train,
              batch_size=128,
              epochs=1,
              validation_data=(x_test, y_test))
        loss.append(hist.history['loss'][0])
        json_string = model.to_json()  
        open('lstm_model_architecture.json','w').write(json_string)  
        model.save_weights('lstm_model_weights.h5') 
    results = pandas.DataFrame()
    results['lstm']=loss
    results.to_csv('lstm_loss.csv')
# first, prepare text samples and their labels
print('Processing text dataset')
######preparing_data###################
texts = []  # list of text samples
labels = []  # list of label ids
data=open(os.path.join(BASE_DIR,'title5month_cms.txt'),'r',encoding='utf-8',errors='ignore')
channel_cons=u'娱乐'
for line in data:
    line=re.sub(' ','',line)
    line=re.sub('，',',',line)
    #line=re.sub(r'[,|【|】|《|》|！|？|?]','',line)
    line=line.strip().split('$$$$') # split the tokens
    if len(line)==7:
        title_text=line[2]
        channel=line[3]
        tmp_label=line[6]
    if len(line)==7 and channel==channel_cons:       
        labels.append(int(tmp_label))
        texts.append(title_text)
    else:
        continue
data.close()

print('Found %s titles.' % len(texts))
############################################
title_token=[]
word_counts = collections.Counter()
for line in texts:
    token=jieba.cut_for_search(line)
    title=','.join(token)
    title_token.append(title)
    word_counts.update(title.strip().split(','))
words_index = {}
for index,word in enumerate(word_counts):
    words_index[word]=index
##############################################
model = word2vec.Word2Vec.load(os.path.join(BASE_DIR,"sina.zh.text.model"))
sequence_vecs=numpy.zeros((len(title_token),MAX_SEQUENCE_LENGTH))
for index,line in enumerate(title_token):
    title=line.strip().split(',')
    sequence = getWordsIndex(title,words_index)
    sequence_vecs[index]=sequence
#########################
num_words = min(MAX_NB_WORDS, len(words_index))
embedding_matrix = numpy.zeros((num_words, EMBEDDING_DIM))
for i,word in enumerate(words_index):
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = model[word].reshape((1,EMBEDDING_DIM))
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except KeyError:
            continue
# words not found in embedding index will be all-zeros.
        
##############    

#label_vecs=numpy.zero((len(labels),2))
#for index,item in enumerate(labels):
x_train, x_test, y_train, y_test=train_test_split(sequence_vecs, labels, test_size=0.2)   

print('Shape of data tensor:', sequence_vecs.shape)
###################
print('Training model.')
# train a lstm
train_cnn(num_words,embedding_matrix)
train_lstm(num_words,embedding_matrix)