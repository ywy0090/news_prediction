# -*- coding: utf-8 -*-

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
from keras.models import model_from_json
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from pandas import DataFrame
import keras.backend as K


BASE_DIR = ''
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 1000000
EMBEDDING_DIM = 400
VALIDATION_SPLIT = 0.2
TRAIN_EPOCHS = 100


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


def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    print('AUC: %f' % roc_auc)

    
########################
def recall(y_true, y_pred):
    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)
    num_tn = K.sum((1.0-y_true)*(1.0-y_pred))
    #print num_tp, num_fn, num_fp, num_tn
    recall =num_tp/(num_tp+num_fn)
    #precision=num_tp/(num_tp+num_fp)
    return recall
 
########loss analysis
results = DataFrame()
p= DataFrame()
results=pandas.read_csv('.csv')
p['lstm']=results['lstm']
results=pandas.read_csv('.csv')
p['cnn']=results['cnn']
p.plot()
plt.show()
########################################
######preparing_data###################
texts = []  # list of text samples
labels = []  # list of label ids
data=open(os.path.join(BASE_DIR,'.txt'),'r',encoding='utf-8',errors='ignore')
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
model = word2vec.Word2Vec.load(os.path.join(BASE_DIR,""))
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

########################################
model = model_from_json(open('lstm_model_architecture.json').read())  
model.load_weights('lstm_model_weights.h5')  
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',recall])
model.fit(x_train, y_train,
              batch_size=128,
              epochs=1,
              validation_data=(x_test, y_test))
y_score_lstm = model.predict(x_test)
fpr_lstm, tpr_lstm, _ = roc_curve(y_test, y_score_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
########################################
model = model_from_json(open('cnn_model_architecture.json').read())  
model.load_weights('cnn_model_weights.h5')
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',recall])
model.fit(x_train, y_train,
              batch_size=128,
              epochs=1,
              validation_data=(x_test, y_test)) 
print('Predicting on test data')
y_score_cnn = model.predict(x_test)

print('Generating results')
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_score_cnn)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
plt.figure()
plt.plot(fpr_cnn, tpr_cnn, label='CNN ROC curve (area = %0.2f)' % roc_auc_cnn,lw=2)
plt.plot(fpr_lstm, tpr_lstm, label='LSTM ROC curve (area = %0.2f)' % roc_auc_lstm,lw=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='best')
plt.show()
print('AUC: %f' % roc_auc_cnn)
print('AUC: %f' % roc_auc_lstm)


