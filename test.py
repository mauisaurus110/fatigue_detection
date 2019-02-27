# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:27:48 2019

@author: Mauisaurus
"""

import face_detection
import cv2
import time

capture = cv2.VideoCapture(0)
i = 0
j = 0

time_start = time.time()
while True:
    ret, frame = capture.read()
    if ret:
        faces = face_detection.run(frame)
        if len(faces):
            (x,y,w,h) = faces[0]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            j = j + 1
        cv2.imshow('frame', frame)
        i = i + 1
    if cv2.waitKey(1) == ord('q'):
        break;
time_end = time.time()

fps = i / (time_end - time_start)
print(j, i,  fps)
capture.release()
cv2.destroyAllWindows()