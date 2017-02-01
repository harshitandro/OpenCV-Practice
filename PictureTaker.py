import cv2
import os
import numpy as np

cascadePath = "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer=cv2.face.createLBPHFaceRecognizer()
video_input=cv2.VideoCapture()

def open_video(target):
    video_input.open(target)
    return video_input.isOpened()

def save_image(subject_name,idno,number,ext,image,location="/home/harshitandro/AutomaticTraining"):
    cv2.imwrite("{}/{}.{}_{}{}".format(location,subject_name,idno,number,ext),image)

def create_face_recog_database(images,labels,subject_name,idno,location="/home/harshitandro/AutomaticTraining/DataSet/"):
    print("Creating face dataset now...")
    recognizer.train(images,np.array(labels))
    recognizer.save('{}{}.{}.yml'.format(location,subject_name,idno))
    print("Face dataset successfully created.")

if __name__ == "__main__":
    if not open_video(0):
        print("Error opening video source")
    subject_name=input("Enter the subject name :")
    idno=int(input("Enter the ID for the given name :"))
    images=[]
    labels=[]
    frame=frame_equalised=None
    total_count=0
    ret=0
    while True:
        ret, frame=video_input.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #clahe = cv2.createCLAHE()
        #frame_equalised=clahe.apply(frame)
        #frame_equalised=cv2.equalizeHist(frame)
        frame_equalised=frame
        detected_face=faceCascade.detectMultiScale(
            frame_equalised,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30),
            flags = 0
        )
        for (x,y,w,h) in detected_face:
            images.append(frame_equalised[y: y + h, x: x + w])
            labels.append(idno)
            save_image(subject_name,idno,total_count,'.png',images[len(images)-1])
        cv2.imshow('Picture Taker',frame)
        total_count+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    create_face_recog_database(images,labels,subject_name,idno)
    video_input.release()