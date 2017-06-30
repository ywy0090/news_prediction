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
from tkinter import *

title=[]
label=[]

#load training data
data=open('','r',encoding='utf-8',errors='ignore')

for line in data:
    line=re.sub(' ','',line)    
    line=re.sub('，',',',line)
    line=line.strip().split('&#') # split the tokens
    title_text=line[0]
    title_text=re.sub('[0-9A-Za-z]','', title_text)
    uv=float(line[1])   
    if uv<=100 and uv>=10:
        label.append(0)
        title.append(title_text)
    elif uv>=1000:
        label.append(1)
        title.append(title_text)
    else:
        continue
data.close()

title_token=[]

for line in title:
    token=jieba.cut_for_search(line)
    title_token.append(', '.join(token))

x_train, x_test, y_train, y_test = train_test_split(title_token, label, test_size=0.2)

tfidftransformer = TfidfTransformer() 

count_v1= CountVectorizer() 
counts_train = count_v1.fit_transform(x_train) 
print("the shape of train is "+repr(counts_train.shape))
tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)

#count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_) 
#counts_test = count_v2.fit_transform(x_test) 
#print("the shape of test is "+repr(counts_test.shape)) 
#tfidf_test = tfidftransformer.fit(counts_test).transform(counts_test)

test_title=' '
test_title_token=[]

token=jieba.cut_for_search(test_title)
test_title_token.append(', '.join(token))

count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_) 
counts_test = count_v2.fit_transform(test_title_token) 
print("the shape of test is "+repr(counts_test.shape)) 
tfidf_test = tfidftransformer.fit(counts_test).transform(counts_test)




word = count_v1.get_feature_names()
#print(word)
model=LogisticRegression()
#model=nb.MultinomialNB(alpha = 0.01)
#model=SGDClassifier(loss='log')
model.fit(tfidf_train, y_train)
pred=model.predict_proba(tfidf_test)
print(pred[:,1])


#pred=model.predict_proba(tfidf_test)
#
##cm=confusion_matrix(y_test, pred)
#
##print(cm)
#
#fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])
#roc_auc = auc(fpr, tpr)
#ks=max(tpr-fpr)
#print ("ks value:%0.3f" %ks) 
##画ROC曲线  
#plt.plot(fpr, tpr, 'k--', color=(0.1, 0.1, 0.1), label='ROC (area = %0.2f)' % roc_auc, lw=2)  
#plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random Chance')   
#plt.xlim([0, 1])  
#plt.ylim([0, 1])  
#plt.xlabel('False Positive Rate')  
#plt.ylabel('True Positive Rate')  
#plt.title('Model ROC curve and ROC area')  
#plt.legend(loc="lower right")  
#plt.show()
