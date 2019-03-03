# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:05:36 2019

@author: Mauisaurus
"""

from sklearn import svm
from sklearn.externals import joblib

data_path = 'train_data0.txt'

#首次训练运行即可
"""
clf = svm.SVC(C = 1, kernel = 'linear', gamma = 20, decision_function_shape = 'ovo')
joblib.dump(clf, "EAR_svm.m")
"""

txt = open(data_path, 'r')
clf = joblib.load("EAR_svm.m")
train = []
labels = []
line_ctr = 0
for txt_str in txt.readlines():
    temp = []
    datas = txt_str.strip()
    datas = datas.split(',')
    label = int( datas.pop() )
    for data in datas:
        data = float(data)
        temp.append(data)
    train.append(temp)
    labels.append(label)
clf.fit(train, labels)
joblib.dump(clf, "EAR_svm.m")

txt.close()