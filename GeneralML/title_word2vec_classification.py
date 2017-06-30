# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 19:54:13 2017

@author: wenyu6
"""
import jieba
import re
import jieba.posseg as pseg
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split  
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import sklearn.naive_bayes as nb
from gensim.models import word2vec
import numpy


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
    vecs = vecs.reshape((vec_len*dimension_len))
    return numpy.array(vecs, dtype='float')

#main
title=[]
label=[]

#load training data
data=open('article_data_auto_tmp2.txt','r',encoding='utf-8',errors='ignore')

for line in data:
    line=re.sub(' ','',line)    
    line=re.sub('，',',',line)
    line=line.strip().split('&#') # split the tokens
    title_text=line[0]
    title_text=re.sub('[0-9A-Za-z]','', title_text)
    uv=float(line[1])   
    if uv<=50 and uv>=10:
        label.append(0)
        title.append(title_text)
    elif uv>=1000:
        label.append(1)
        title.append(title_text)
    else:
        continue
data.close()
##
title_token=[]

for line in title:
    token=jieba.cut_for_search(line)
    title_token.append(', '.join(token))
#prepare vecs
#fileSegWordDonePath ='article_title_seg.txt'
word2vec_vecs=[]
#sentences = word2vec.Text8Corpus(title_token)
#sentences= [s.split() for s in title_token]
model = word2vec.Word2Vec.load("sina.zh.text.model")
#model=word2vec.Word2Vec(sentences, size=300, min_count=1)
#with open(fileSegWordDonePath,'r',encoding='utf-8') as fr:
#print(model[u'中国'])
for line in title_token:
    title=line.strip().split(',')
    tmp_vec = getTitleVecs(title)
    word2vec_vecs.append(tmp_vec)

x_train, x_test, y_train, y_test=train_test_split(word2vec_vecs, label, test_size=0.2)
#model 
model=LogisticRegression()
model.fit(x_train, y_train)
#pred=model.predict_proba(y_test[0:3000])
#print(pred[:,1])
#pred=model.predict_proba(x_test)
pred=model.predict_proba(x_test)
#cm=confusion_matrix(y_test, pred)
#print(cm)
fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])
roc_auc = auc(fpr, tpr)
ks=max(tpr-fpr)
print ("ks value:%0.3f" %ks) 
#画ROC曲线  
plt.plot(fpr, tpr, 'k--', color=(0.1, 0.1, 0.1), label='ROC (area = %0.2f)' % roc_auc, lw=2)  
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random Chance')   
plt.xlim([0, 1])  
plt.ylim([0, 1])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Model ROC curve and ROC area')  
plt.legend(loc="lower right")  
plt.show()