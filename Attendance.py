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


enco  = findEncodings(images)
print('Encoding Complete')
# Step 3: initialize the webcam
# #detect the face   face location printes out 4 values Top, Right ,Bottom ,Left 
# faceLoc = face_recognition.face_locations(imgElon)[0]
# #encode the face we have detacted 
# encodeElon = face_recognition.face_encodings(imgElon)[0]

# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# # detect the face location printes out 4 values Top, Right ,Bottom ,Left 
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# #encode the face we have detacted 
# encodeTest = face_recognition.face_encodings(imgTest)[0]

# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# # Comparing the faces and finding the distance between them

# results = face_recognition.compare_faces([encodeElon],encodeTest)
# #find the distance to get the best match
# faceDist = face_recognition.face_distance([encodeElon],encodeTest)
# print(results, faceDist)



