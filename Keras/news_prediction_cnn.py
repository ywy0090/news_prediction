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
import os
from io import open
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.cross_validation import train_test_split  
from keras.models import Sequential
from keras.layers import Dropout

BASE_DIR = 'D:\\python_project\\news_prediction\\'
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 400
VALIDATION_SPLIT = 0.2

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
for line in texts:
    token=jieba.cut_for_search(line)
    title=','.join(token)
    title_token.append(title)
############################################
model = word2vec.Word2Vec.load(os.path.join(BASE_DIR,"sina.zh.text.model"))
word2vec_vecs=numpy.zeros((len(title_token), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
for index,item in enumerate(title_token):
    title=line.strip().split(',')
    tmp_vec = getTitleVecs(title)
    word2vec_vecs[index]=tmp_vec

#label_vecs=numpy.zero((len(labels),2))
#for index,item in enumerate(labels):
x_train, x_test, y_train, y_test=train_test_split(word2vec_vecs, labels, test_size=0.2)   

print('Shape of data tensor:', len(word2vec_vecs))
###################
print('Training model.')
# train a 1D convnet with global maxpooling
filters=250
model = Sequential()
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

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_test, y_test))