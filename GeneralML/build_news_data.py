# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:59:27 2017

@author: wenyu6
"""


import jieba
import re

#content=[]
##rootdir=""
#rootdir = ""
#save_final = []
#f_d = open('content'+'_mergeRes.txt', 'w')
#for parent,dirnames,filenames in os.walk(rootdir):      
#    xml_listdir = os.path.join(parent, single_d)
#    for f in os.listdir(xml_listdir):
#        f_s = open((xml_listdir+'\\'+f),'r')
#        f_d.write(f_s.read())
#f_d.close()
#load training data
content_token=[]    
data=open('content_mergeRes.txt','r', encoding='utf-8',errors='ignore')
seg_data=open('content_SegRes.txt','w',encoding='utf-8',errors='ignore')
#user defined dict
jieba.load_userdict("tags5month_cms.txt")
n=0
for line in data:
    n=n+1
    if n%10000==0:
        print(n)
    line=re.sub(' ','',line)  
    line=re.sub('font-family:宋体','',line)
    line=re.sub('（文：','',line)
    line=re.sub('，',',',line)
    line=re.sub('[^\u4e00-\u9fa5]','',line)
    line=line.strip()  # split the tokens
    token=jieba.cut_for_search(line)
    seg_data.write(' '.join(token))
    seg_data.write('\n')
    
data.close()
seg_data.close()



#for line in content:
#    token=jieba.cut_for_search(line)
#    content_token.append(', '.join(token))

