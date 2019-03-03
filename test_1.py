# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:47:14 2019

@author: Mauisaurus
"""
import cv2
from scipy.spatial import distance
from imutils import face_utils
import face_detection
import shape_prediction
from sklearn.externals import joblib

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

clf = joblib.load("EAR_svm.m")
print(type(clf))
video_path = 'train_video0.mp4'
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1
FRAME_COUNT = 6

frames = []
frames_EAR = []
capture = cv2.VideoCapture(video_path)
vector_size = 2 * FRAME_COUNT + 1
while True:
    ret, frame = capture.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.run(gray)
        if len(faces):
            shape = shape_prediction.run(gray, faces[0])
            points = face_utils.shape_to_np(shape)
            for index, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                cv2.circle(frame, pt_pos, 2, (0, 0, 255), 1)
            leftEye = points[LEFT_EYE_START: LEFT_EYE_END + 1]
            rightEye = points[RIGHT_EYE_START: RIGHT_EYE_END + 1]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            EAR = (leftEAR + rightEAR) / 2.0
            #frames.append(frame)
            frames_EAR.append(EAR)
            #able = able + 1
        else:
            #frames.clear()
            frames_EAR.clear()

        if len(frames_EAR) == vector_size:
            temp = []
            temp.append(frames_EAR)
            res = clf.predict(temp)
            print(res)
            """
            i = 0
            while True:
                frame = frames[i % vector_size].copy()
                cv2.putText(frame, "res:{0}".format(res[0]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame, "Blinks:{0}".format(i % vector_size + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow('frame', frame)
                i = i + 1
                key = cv2.waitKey(0)
                if key == ord('n'):
                    break
            frames.pop(0)
            """
            frames_EAR.pop(0)
                    
        #count = count + 1
    else:
        break

capture.release()
