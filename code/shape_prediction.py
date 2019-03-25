# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:42:40 2019

@author: Mauisaurus
"""

import dlib

predictor = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat')# 人脸特征点检测器

def run(image, box):
    return predictor(image, box)# 检测特征点
