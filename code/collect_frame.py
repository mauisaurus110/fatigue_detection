# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:01:03 2019

@author: Mauisaurus
"""
import cv2
from scipy.spatial import distance
from imutils import face_utils
import face_detection
import shape_prediction

video_path = '../frame_data/wp.mp4'
data_path = '../EAR_data/wp.txt'
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1
FRAME_COUNT = 6
vector_size = 2 * FRAME_COUNT + 1

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

txt = open(data_path, 'w+')
capture = cv2.VideoCapture(video_path)
count = 0
able = 0
frames = []
frames_EAR = []
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
            frames.append(frame)
            frames_EAR.append(EAR)
            able = able + 1
        else:
            frames.clear()
            frames_EAR.clear()     
        if len(frames) == vector_size:
            i = 0
            while True:
                frame = frames[i % vector_size].copy()
                cv2.putText(frame, "{0}".format(i % vector_size + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow('frame', frame)
                i = i + 1
                key = cv2.waitKey(0)
                if key == ord('y'):
                    label = 0
                    break
                elif key == ord('n'):
                    label = 1
                    break
            for j in frames_EAR:
                txt.write(str(j))
                txt.write(',')
            txt.write(str(label))    
            txt.write('\n')
            frames.pop(0)
            frames_EAR.pop(0)
        count = count + 1
    else:
        break

print(able, count)
txt.close()
capture.release()
cv2.destroyAllWindows()