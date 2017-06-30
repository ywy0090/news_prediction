# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:22:03 2017

@author: wenyu6
"""

import jieba
import re'
import os
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier


BASE_DIR = 'D:\\python_project\\news_prediction\\'
title=[]
label=[]
uvs=[]
#load training data
data=open(os.path.join(BASE_DIR,'title5month_cms.txt'),'r',encoding='utf-16',errors='ignore')
channel_cons=u'体育'
for line in data:
    line=re.sub(' ','',line)
    line=re.sub('，',',',line)
    line=re.sub(r'[,|【|】|《|》|！|？|?]','',line)
    line=line.strip().split('$$$$') # split the tokens
    if len(line)==7:
        title_text=line[2]
#        num=re.findall(r'\d{1,}',title_text)
#        if(len(num)>0):
#            for i in range(0,len(num)):
#                title_text = title_text.replace(str(num[i]),str(len(num[i])))
#                #print(title_text)    
        channel=line[3]
        tmp_label=line[6]
        tmp_uv=line[5]
        try:
            uv =int(tmp_uv)
        except Exception:
            continue
    #print(channel)
#    try:
#        uv=float(line[2])
#    except Exception:
#        print(line[2])
    if uv>=5 and uv<=200 and channel==channel_cons:
        label.append(0)
        title.append(title_text)
#    if len(line)==7 and channel==channel_cons :       
#        label.append(int(tmp_label))
#        title.append(title_text)
#        uvs.append()
    elif uv>=2000 and channel==channel_cons:
        label.append(1)
        title.append(title_text)
    else:
        continue
data.close()

#user defined dict
#jieba.load_userdict("tags5month_cms.txt")
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
x_train, x_test, y_train, y_test = train_test_split(tfidf, label, test_size=0.2, random_state=10)
#model=ensemble.GradientBoostingClassifier(n_estimators=100)
#model = AdaBoostClassifier(LogisticRegression(penalty='l2', C=.0001,class_weight='balanced'),
#                        algorithm="SAMME",
#                         n_estimators=100)
#model=SGDClassifier(loss='log',class_weight='balanced')
model=LogisticRegression(penalty='l2',class_weight='balanced',C=0.00001)
model.fit(x_train, y_train)
###
pred=model.predict(x_train)
#pred=model.predict(x_test.toarray())
acc_rate=accuracy_score(y_train, pred)
recall_rate=recall_score(y_train, pred)
precision_rate=precision_score(y_train, pred)

print("trainning accuracy is %f"%acc_rate)
print("trainning recall is %f"%recall_rate)
print("trainning precision is %f"%precision_rate)
####
pred=model.predict(x_test)
#pred=model.predict(x_test.toarray())
acc_rate=accuracy_score(y_test, pred)
recall_rate=recall_score(y_test, pred)
precision_rate=precision_score(y_test, pred)

print("testing accuracy is %f"%acc_rate)
print("testing recall is %f"%recall_rate)
print("testing precision is %f"%precision_rate)

#####
cm=confusion_matrix(y_test, pred)
#
print(cm)
#
pred = model.predict_proba(x_test)
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