# -*- coding: utf-8 -*-

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
from sklearn.metrics import accuracy_score, recall_score,precision_score
import numpy

def buildWordVector(text, size):
    vec = numpy.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


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
data=open('.txt','r',encoding='utf-8',errors='ignore')
channel_cons=u''
for line in data:
    line=re.sub(' ','',line)
    line=re.sub('，',',',line)
    #line=re.sub(r'[,|【|】|《|》|！|？|?]','',line)
    line=line.strip().split('$$$$') # split the tokens
    if len(line)==7:
        title_text=line[2]
    #print(title_text)
        channel=line[3]
        tmp_label=line[6]
    #print(channel)
#    try:
#        uv=float(line[2])
#    except Exception:
#        print(line[2])
#    if uv>=10 and uv<=100 and channel==channel_cons:
    if len(line)==7 and channel==channel_cons:       
        label.append(int(tmp_label))
        title.append(title_text)
#    elif uv>=2000 and channel==channel_cons:
#        label.append(1)
#        title.append(title_text)
    else:
        continue
data.close()
#for line in data:
#    line=re.sub(' ','',line)
#    line=re.sub('，',',',line)
#    line=line.strip().split('$$$') # split the tokens
#    title_text=line[0]
#    #print(title_text)
#    channel=line[1]
#    #print(channel)
#    try:
#        uv=float(line[2])
#    except Exception:
#        print(line[2])
#    if uv>=10 and uv<=100 and channel==channel_cons:
#        label.append(0)
#        title.append(title_text)
#    elif uv>=2000 and channel==channel_cons:
#        label.append(1)
#        title.append(title_text)
#    else:
#        continue
#data.close()
##
title_token=[]

for line in title:
    token=jieba.cut_for_search(line)
    title_token.append(', '.join(token))
#prepare vecs
#fileSegWordDonePath ='.txt'
word2vec_vecs=[]
#sentences = word2vec.Text8Corpus(title_token)
#sentences= [s.split() for s in title_token]
model = word2vec.Word2Vec.load("")
#model=word2vec.Word2Vec(sentences, size=300, min_count=1)
#with open(fileSegWordDonePath,'r',encoding='utf-8') as fr:
#print(model[u''])
for line in title_token:
    title=line.strip().split(',')
    tmp_vec = getTitleVecs(title)
    word2vec_vecs.append(tmp_vec)

x_train, x_test, y_train, y_test=train_test_split(word2vec_vecs, label, test_size=0.2)
####
#model 
model=LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)
#pred=model.predict_proba(y_test[0:3000])
#print(pred[:,1])
#pred=model.predict_proba(x_test)
pred=model.predict_proba(x_test)
pred=model.predict(x_test)
acc_rate=accuracy_score(y_test, pred)
recall_rate=recall_score(y_test, pred)
precision_rate=precision_score(y_test, pred)

print("accuracy is %f"%acc_rate)
print("recall is %f"%recall_rate)
print("precision is %f"%precision_rate)
cm=confusion_matrix(y_test, pred)
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
