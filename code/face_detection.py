# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:39:11 2019

@author: Mauisaurus
"""

"""
import cv2

path_classifier = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path_classifier)
scaleFactor = 1.5
minNeighbors = 3

def run(frame):
    return face_cascade.detectMultiScale(frame, scaleFactor, minNeighbors)

"""
import dlib

detector = dlib.get_frontal_face_detector()

def run(frame):
    return detector(frame, 0)
