
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

def buildWordVector(text, size):
    vec = numpy.zeros(size).reshape((1, size))
    count = 0.
    text=text.strip().split(',')
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

    
#main
title=[]
label=[]

#load training data
data=open('','r',encoding='utf-8',errors='ignore')
channel_cons=u''
for line in data:
    line=re.sub(' ','',line)
    line=re.sub('，',',',line)
    line=line.strip().split('$$$') # split the tokens
    title_text=line[0]
    #print(title_text)
    channel=line[1]
    #print(channel)
    try:
        uv=float(line[2])
    except Exception:
        print(line[2])
    if uv>=10 and uv<=100 and channel==channel_cons:
        label.append(0)
        title.append(title_text)
    elif uv>=2000 and channel==channel_cons:
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

train_vecs = numpy.concatenate([buildWordVector(z, 400) for z in title_token])


x_train, x_test, y_train, y_test=train_test_split(train_vecs, label, test_size=0.2)
####
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
