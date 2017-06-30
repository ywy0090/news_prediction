# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:20:33 2017

@author: wenyu6
"""
import os
import re
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt


BASE_DIR = 'D:\\python_project\\news_prediction\\'


def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

# first, prepare text samples and their labels
print('Processing text dataset')
######preparing_data###################
dts=[]
texts = []  # list of text samples
labels = []  # list of label ids
channels=[] #list of channels
uvs=[]
data=open(os.path.join(BASE_DIR,'title5month_cms.txt'),'r',encoding='utf-16',errors='ignore')
#data=open(os.path.join(BASE_DIR,'title5month_cms.txt'),'r',encoding='utf-8')
data.flush()

for line in data:
    line=re.sub(' ','',line)
    line=re.sub('，',',',line)
    #line=re.sub(r'[,|【|】|《|》|！|？|?]','',line)
    line=line.strip().split('$$$$') # split the tokens
    if len(line)==7:
        tmp_dt=line[0]
        title_text=line[2]
        channel=line[3]
        tmp_uv=line[5]
        tmp_label=line[6]
        try:
            tmp_uv =int(tmp_uv)
            tmp_label=int(tmp_label)
        except Exception:
            continue
    if len(line)==7 and(tmp_label==1 or tmp_label==0) :
        dts.append(tmp_dt)
        labels.append(int(tmp_label))
        channels.append(channel)
        uvs.append(tmp_uv)
        texts.append(title_text)
        
    else:
        continue
data.close()
print('Found %s titles.' % len(texts))

###
results = DataFrame()
out = DataFrame()
results['uvs']=uvs
results['channels']=channels
results['dts']=dts
col=results['channels']
ent=results[results['channels'].isin([channel_cons])]
results.groupby(results['channels']).mean()
results.groupby(results['channels']).min()
results.groupby(results['channels']).max()
results.groupby(results['channels']).std()
out = results.groupby(results['channels']).describe()
out=results.groupby(['channels','dts']).describe()
out=results.groupby(['channels','dts']).perc()

factor=pd.cut(ent.uvs,10)
out.to_csv('channel_describe.csv')
grouped = ent.uvs.groupby(factor)
grouped.apply(get_stats).unstack()
########channel binning
channel_cons=u'体育'
ent=results[results['channels'].isin([channel_cons])]
#ent=ent[ent>3]
factor=pd.qcut(ent.uvs,10, labels=False)
grouped = ent.uvs.groupby(factor)
grouped.apply(get_stats).unstack()
#p.plot()
#plt.show()