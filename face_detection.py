# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:39:11 2019

@author: Mauisaurus
"""

import cv2

path_classifier = 'C:\\Users\\lenovo\\Desktop\\biye\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path_classifier)
scaleFactor = 1.5
minNeighbors = 3

def run(frame):
    return face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor, minNeighbors)
    
