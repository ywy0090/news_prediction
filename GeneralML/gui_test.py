# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 16:14:51 2017

@author: wenyu6
"""

#coding:utf-8  
#!/usr/bin/python  
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
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import sklearn.naive_bayes as nb
  
from tkinter import Tk,Label,Entry,StringVar,Button  
from tkinter import BOTTOM
from tkinter import messagebox
  
#　点击button时对应的操作  
def on_hello():
    print('hello')


def on_click(count_v1, model):
    title_str = title_text.get()
    popular_prob = predict(title_str, count_v1, model)
    string = str("新闻标题：%s，其流行概率为%s" %(title_str,popular_prob))
    messagebox.showinfo(title='预测结果', message = string) 
    

def predict(title_str, count_v1, model):
    test_title_token=[]
    token=jieba.cut_for_search(title_str)
    test_title_token.append(', '.join(token))
    count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_) 
    counts_test = count_v2.fit_transform(test_title_token) 
    tfidf_test = tfidftransformer.fit(counts_test).transform(counts_test)
    pred=model.predict_proba(tfidf_test)
    return pred[:,1]

# 次级窗口  
############model  
 
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
    if uv<=100 and uv>=10:
        label.append(0)
        title.append(title_text)
    elif uv>=1000:
        label.append(1)
        title.append(title_text)
    else:
        continue
data.close()
##tokenize
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
##
word = count_v1.get_feature_names()
#print(word)
model=LogisticRegression()
#model=nb.MultinomialNB(alpha = 0.01)
#model=SGDClassifier(loss='log')
model.fit(tfidf_train, y_train)
#事件循环
#　根窗口  
root = Tk()   
root.title('window with command') #主窗口标题  
root.geometry('400x200')  #主窗口大小，中间的为英文字母x  
#
l1 = Label(root, text="新闻标题：")
l1.pack()  #这里的side可以赋值为LEFT  RTGHT TOP  BOTTOM
title_text = StringVar()
title_entry = Entry(root, textvariable = title_text)
title_text.set(" ")
title_entry.pack()  
com = Button(root,text = '预测', command = lambda : on_click(count_v1, model)) 
#　第一个参数root说明com按钮是root的孩子，text指按钮的名称，command指点击按钮时所执行的操作  
#com = Button(root,text = '预测', command = on_hello)
com.pack(side = BOTTOM)  #　次级窗口的位置摆放位置  
root.mainloop()
## 
