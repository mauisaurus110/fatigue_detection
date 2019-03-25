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

clf = joblib.load('../model/EAR_svm.m')
video_path = '../frame_data/myl.mp4'
out_path = '../output/result.mp4'
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1
FRAME_COUNT = 6

frames_EAR = []
frames_EAR.append(0.3)
capture = cv2.VideoCapture(video_path)
capture.set(cv2.CAP_PROP_FPS, 30)
vout = cv2.VideoWriter()
sz = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vout.open(out_path, cv2.VideoWriter_fourcc(*'mpeg'), capture.get(cv2.CAP_PROP_FPS), sz, True)
        
count = 0
blink_count = 0
last_blink = -7
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
            frames_EAR.append(EAR)
        else:
            frames_EAR.append(frames_EAR[-1])
        
        if len(frames_EAR) == vector_size:
            temp = []
            temp.append(frames_EAR)
            res = clf.predict(temp)
            #眨眼间隔小于7无效
            if res[0] == 0:
                if count - last_blink >= 7:
                    blink_count = blink_count + 1
                last_blink = count
            frames_EAR.pop(0)

        cv2.putText(frame, "Blinks:{0}".format(blink_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)         
        vout.write(frame)
        count = count + 1
    else:
        break

capture.release()
vout.release()