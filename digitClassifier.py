import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
print('Starting.................')
myList = os.listdir(path="data")
print('Number of Classes=',len(myList))
images=[]
classes=[]
myList.sort()
print('Importing Classes.........')
for i in myList:
    path='data'+'/'+str(i)
    imgList = os.listdir(path)
    for j in imgList:
        img = cv2.imread(path+'/'+str(j))
        img = cv2.resize(img,(32,32))
        images.append(img)
        classes.append(i)
    print(i,end=" ")
print(' ')
print('Number of Images imported = ',len(images))
# cv2.imshow("output",images[0])
# cv2.waitKey(0)
X_train,X_test,Y_train,Y_test = train_test_split(images,classes,test_size=0.2,random_state=5)
print('Size of Training data = ', len(X_train))