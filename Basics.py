import cv2
import numpy as np
import face_recognition

# get the encoding 
#get our image

imgElon = face_recognition.load_image_file('imagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('imagesBasic/Elon Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#detect the face   face location printes out 4 values Top, Right ,Bottom ,Left 
faceLoc = face_recognition.face_locations(imgElon)[0]
#encode the face we have detacted 
encodeElon = face_recognition.face_encodings(imgElon)[0]

cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
