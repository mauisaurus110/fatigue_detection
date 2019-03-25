# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:05:36 2019

@author: Mauisaurus
"""

import os
from sklearn import svm
from sklearn.externals import joblib

data_path = '../EAR_data'
svm_path = '../model/EAR_svm.m'

files = os.listdir(data_path)
txts = []
for file in files:
    txts.append(open(data_path + '/' + file, 'r'))
    
clf = svm.SVC(C = 10, kernel = 'linear', gamma = 20, decision_function_shape = 'ovo')
train = []
labels = []
for txt in txts:
    for txt_str in txt.readlines():
        datas = txt_str.strip()
        datas = datas.split(',')
        label = int( datas.pop() )
        temp = []
        for data in datas:
            temp.append(float(data))
        train.append(temp)
        labels.append(label)

clf.fit(train, labels)
joblib.dump(clf, svm_path)

for txt in txts:
    txt.close()