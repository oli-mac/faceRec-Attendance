from sre_constants import SUCCESS
from unittest import result
import cv2
import numpy as np
import face_recognition
import os

#step 1: import images and conver them to RGB
path = 'imagesAttendance'
images = []
#take the names from the image it self
classNames =[]

myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
#step 2: encode the faces 

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown  = findEncodings(images)
print('Encoding Complete')
# Step 3: initialize the webcam

cap = cv2.VideoCapture(0)

while True:
    success, img =cap.read()
    #reduse size of image
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)
#we might find multiple faces on the web cam for that we calcualte the location of the faces
    faceCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, faceCurFrame)

# Step 4: finding matches
    for encodeFace, FaceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDistance)
        matchIndex = np.argmin(faceDistance)

# Step 5: display bounding box and write the name     
        if matches[matchIndex]:
            name = classNames[matchIndex]
            # print(name)
            y1,x2,y2,x1 = FaceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('Webcam',img)
    cv2.waitKey(0)

# Step 5: Mark the Attendance





