# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:47:14 2017

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
from sklearn.metrics import accuracy_score, recall_score,precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
import numpy

title=[]
label=[]

#load training data
data=open('title5month_cms.txt','r',encoding='utf-8',errors='ignore')
channel_cons=u'娱乐'
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

#user defined dict
jieba.load_userdict("tags5month_cms.txt")
title_token=[]

for line in title:
    token=jieba.cut_for_search(line)
    title_token.append(', '.join(token))
#print(title_token)
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(title_token))
word = vectorizer.get_feature_names()
count_v = vectorizer.fit_transform(title_token)

x_train, x_test, y_train, y_test = train_test_split(tfidf, label, test_size=0.2)
x_train, x_train_lr, y_train, y_train_lr = train_test_split(x_train, y_train, test_size=0.2)
#model=ensemble.GradientBoostingClassifier()
n_estimator=10
# Unsupervised transformation based on totally random trees
#numpy.random.seed(10)
#rt = RandomTreesEmbedding( n_estimators=n_estimator,
#    random_state=0)
#rt_lm = LogisticRegression()
#pipeline = make_pipeline(rt, rt_lm)
#pipeline.fit(x_train, y_train)
#y_pred_rt = pipeline.predict(x_test)
#gbdt+lr
grd = ensemble.GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(x_train, y_train)
grd_enc.fit(grd.apply(x_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(x_train_lr)[:, :, 0]), y_train_lr)
#
#y_pred_grd_lm = grd_lm.predict(
#    grd_enc.transform(grd.apply(x_test)[:, :, 0]))

#fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

#fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

#model=SGDClassifier(loss='log',class_weight='balanced')
#model=SGDClassifier(loss='log')
#model.fit(x_train, y_train)

#pred=model.predict(x_test)
y_pred_grd = grd.predict(x_test.toarray())
acc_rate=accuracy_score(y_test, y_pred_grd)
recall_rate=recall_score(y_test, y_pred_grd)
precision_rate=precision_score(y_test, y_pred_grd)

print("accuracy is %f"%acc_rate)
print("recall is %f"%recall_rate)
print("precision is %f"%precision_rate)
cm=confusion_matrix(y_test, y_pred_grd_lm)
#
print(cm)
#
#pred = grd_lm.predict_proba(grd_enc.transform(grd.apply(x_test)[:, :, 0]))
pred = grd.predict_proba(x_test.toarray())
fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])

#fpr, tpr, thresholds = roc_curve(y_test, pred)
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