#!/usr/bin/python
import cv2
import sys
import numpy as np

cascadePath = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.createLBPHFaceRecognizer()
webcam =cv2.VideoCapture(0)

def load_trainer(trainingDataSet):
    recognizer.load(trainingDataSet)
    print('Training dataset loaded')

if __name__ == '__main__':
    #dataset=input('Enter location of dataset :')
    dataset="/home/harshitandro/AutomaticTraining/DataSet/hs.123.yml"
    load_trainer(dataset)
    font = cv2.FONT_HERSHEY_PLAIN
    if not webcam.isOpened():
        print('Error opening camera.')
        sys.exit(1)
    flag=0
    while True:
        flag , frame=webcam.read()
        bg_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #frame_equalised=cv2.equalizeHist(bg_frame)
        frame_equalised=bg_frame
        detected_face=faceCascade.detectMultiScale(
            frame_equalised,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30),
            flags = 0
        )
        for (x,y,w,h) in detected_face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            nbr_predicted, conf = 0,0.0
            nbr_predicted, conf = recognizer.predict(frame_equalised[y: y + h, x: x + w])
            cv2.putText(frame,str((nbr_predicted,conf)),(x,y+h),font,2,(0,255,0))
        cv2.imshow('My Recognizer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()