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

# imgElon = face_recognition.load_image_file('imagesBasic/Elon Musk.jpg')
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

# imgTest = face_recognition.load_image_file('imagesBasic/Jeff Test.jpg')
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)








