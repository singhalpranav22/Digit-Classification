import cv2
import numpy as np

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img=img/255 #normalise the grayscale image
    return img