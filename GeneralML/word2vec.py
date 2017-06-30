# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:42:45 2017

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
from tkinter import *
from gensim.models import word2vec
import numpy
import logging
from gensim.models.word2vec import LineSentence
import multiprocessing


    
fileSegPath ='content_SegRes.txt'
#sentences = word2vec.Text8Corpus(fileSegPath)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#model=word2vec.Word2Vec(sentences, size=300, min_count=3)
model=word2vec.Word2Vec(LineSentence(fileSegPath), size=400, window=5,min_count=3,
                        workers=multiprocessing.cpu_count())
# outp1 为输出模型
outp1 = 'sina.zh.text.model'
# outp2为原始c版本word2vec的vector格式的模型
outp2 = 'sina.zh.text.vector' 
model.save(outp1)
model.wv.save_word2vec_format(outp2, binary=False)
print(model[u'熊猫'])
# 计算两个词的相似度/相关程度
y1 = model.similarity(u"不错", u"好")
print (u"【不错】和【好】的相似度为：", y1)
print ("--------\n")

# 计算某个词的相关词列表
y2 = model.most_similar(u"冬天", topn=20)  # 20个最相关的
print (u"和【冬天】最相关的词有：\n")
for item in y2:
    print (item[0], item[1])
print ("--------\n")