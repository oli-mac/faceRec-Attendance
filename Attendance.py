from unittest import result
import cv2
import numpy as np
import face_recognition

#step 1: import images and conver them to RGB

imgElon = face_recognition.load_image_file('imagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('imagesBasic/Jeff Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)








