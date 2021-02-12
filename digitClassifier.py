import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utility import *
print('Starting.................')
myList = os.listdir(path="data")
print('Number of Classes=',len(myList))
noClass = len(myList)
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
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.2,random_state=5)
X_train=np.array(X_train)
X_val=np.array(X_val)
X_test=np.array(X_test)
Y_val=np.array(Y_val)
Y_test=np.array(Y_test)
Y_train=np.array(Y_train)
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
print('Size of Training data = ', len(X_train))

numofSamples = [] # List stores no. of training data for a particular class
for i in range(noClass):
    numofSamples.append(len(np.where(Y_train==str(i))[0]))
print('Number of Samples for a class in Training Set = ',numofSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noClass),numofSamples)
plt.title('Number of Images in the training set for each class.')
plt.xlabel('Class Id')
plt.ylabel('Number of Images')
plt.show()

X_train = np.array(list(map(preprocessing,X_train)))
X_test = np.array(list(map(preprocessing,X_test)))
X_val = np.array(list(map(preprocessing,X_val)))
# Now adding depth to the images for the fine working of the convulation neural network
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_val=X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)
print(X_train.shape)
img = X_train[67]
img = cv2.resize(img,(300,300))
cv2.imshow(str(Y_train[67]),img)
cv2.waitKey(0)