# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:05:50 2017

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.pipeline import make_pipeline
import numpy


def performance_output(y_test, y_pred):
    acc_rate=accuracy_score(y_test, y_pred)
    recall_rate=recall_score(y_test, y_pred)
    precision_rate=precision_score(y_test, y_pred)
    print("accuracy is %f"%acc_rate)
    print("recall is %f"%recall_rate)
    print("precision is %f"%precision_rate)
    cm=confusion_matrix(y_test, y_pred)
    print(cm)    


title=[]
label=[]

#load training data
data=open('','r',encoding='utf-8',errors='ignore')
channel_cons=
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

# Create and fit an AdaBoosted decision tree
#class_weight='balanced'
bdt = AdaBoostClassifier(LogisticRegression(),
                         algorithm="SAMME",
                         n_estimators=100)
bdt.fit(x_train, y_train)
y_pred_bdt = bdt.predict(x_test)
print('AdaBoostClassifier:')
performance_output(y_test,y_pred_bdt)
pred = bdt.predict_proba(x_test)
fpr_bdt, tpr_bdt, _ = roc_curve(y_test, pred[:,1])
# Unsupervised transformation based on totally random trees
n_estimator=100
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
    random_state=0)

rt_lm = LogisticRegression()
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(x_train, y_train)
##random trees+lr calculate rate
y_pred_rt = pipeline.predict(x_test)
print('unsupervised random trees+LR:')
performance_output(y_test,y_pred_rt)
#roc
y_pred_rt = pipeline.predict_proba(x_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf.fit(x_train, y_train)
rf_enc.fit(rf.apply(x_train))
rf_lm.fit(rf_enc.transform(rf.apply(x_train)), y_train)
##random forest+lr calculate rate
y_pred_rf_lm = rf_lm.predict(rf_enc.transform(rf.apply(x_test)))
print('random trees+LR:')
performance_output(y_test,y_pred_rf_lm)
##roc
y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(x_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
##GBDT+lr
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(x_train, y_train)
grd_enc.fit(grd.apply(x_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(x_train)[:, :, 0]), y_train)
#GBDT+lr rate
y_pred_grd_lm = grd_lm.predict(
    grd_enc.transform(grd.apply(x_test)[:, :, 0]))
print('GBDT+LR:')
performance_output(y_test,y_pred_grd_lm)
#GBDT+lr roc                            
y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(x_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
# The gradient boosted model by itself
y_pred_grd = grd.predict(x_test.toarray())
print('GBDT:')
performance_output(y_test,y_pred_grd)
#GBDT roc
y_pred_grd = grd.predict_proba(x_test.toarray())[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
# The random forest model by itself
y_pred_rf = rf.predict(x_test)
print('randomforest:')
performance_output(y_test,y_pred_rf)
#random forest roc
y_pred_rf = rf.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
# Create and fit an AdaBoosted LR
#class_weight='balanced'
bdt_200 = AdaBoostClassifier(LogisticRegression(),
                         algorithm="SAMME",
                         n_estimators=200)
bdt_200.fit(x_train, y_train)
y_pred_bdt = bdt_200.predict(x_test)
print('AdaBoostClassifier200 estimators:')
performance_output(y_test,y_pred_bdt)
pred = bdt_200.predict_proba(x_test)
fpr_bdt_200, tpr_bdt_200, _ = roc_curve(y_test, pred[:,1])
#LR
model=SGDClassifier(loss='log', class_weight='balanced')
model.fit(x_train, y_train)
y_pred_lr = model.predict(x_test)
print('LR estimators:')
performance_output(y_test,y_pred_lr)
pred_lr = model.predict_proba(x_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, pred_lr[:,1])
#画ROC曲线
plt.plot([0, 1], [0, 1], 'k--')
roc_auc = auc(fpr_rt_lm, tpr_rt_lm)
plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR(area = %0.2f)'%roc_auc)
roc_auc = auc(fpr_bdt, tpr_bdt)
plt.plot(fpr_bdt, tpr_bdt, label='AdaBoost(area = %0.2f)'%roc_auc)
roc_auc = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label='LR(area = %0.2f)'%roc_auc)
#plt.plot(fpr_bdt_200, tpr_bdt_200, label='AdaBoost200')
roc_auc = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label='RF(area = %0.2f)'%roc_auc)
roc_auc = auc(fpr_rf_lm, tpr_rf_lm)
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR(area = %0.2f)'%roc_auc)
roc_auc = auc(fpr_grd, tpr_grd)
plt.plot(fpr_grd, tpr_grd, label='GBT(area = %0.2f)'%roc_auc)
roc_auc = auc(fpr_grd_lm, tpr_grd_lm)
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR(area = %0.2f)'%roc_auc)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
