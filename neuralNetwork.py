import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import warnings
import glob
from preprocessing import preprocess
# warnings.filterwarnings('ignore')


# numberOfTextures = 2#47
maxNum = 10
imagearr = []
# for folder in glob.glob("dtd/images/*"):
#     count =0
#     for imageFile in glob.glob(folder+ "/*"):
#         if count >= maxNum:
#             break
#         count +=1
#         # print(imageFile)
#         imagearr.append(cv2.imread(imageFile, flags=cv2.IMREAD_GRAYSCALE))



# image = cv2.imread("dtd/images/banded/banded_0004.jpg", flags=cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image,(256,256))
# print(image)
# plt.imshow(image, cmap='gray')
# plt.show()

x, y, numberOfTextures = preprocess()
# x = preprocess()
# y = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],
#      [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]
#     ])
# plt.imshow(imagearr[13], cmap='gray')
# plt.show()
print(x.shape)
# newX=[]
# for i,img in enumerate(x):
#     print(img.flatten().shape)
#     newX.append(x[i]/255)
# # x = x.flatten()
print("WE DID IT REDDIT")
# newX = np.array(newX)
# print(newX.shape)
# x = newX
# print(y.shape)
# q = tf.keras.utils.normalize(x)
# print(x.shape)
# nx = np.divide(x[0],255)
print(x[0])
model = tf.keras.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(512,512)))#input_shape=(512*512,)
model.add(tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(512*512,)))
model.add(tf.keras.layers.Dense(22, activation='sigmoid'))
model.add(tf.keras.layers.Dense(12, activation='softmax'))
model.add(tf.keras.layers.Dense(numberOfTextures))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(x, y, epochs=10, validation_split=.2)
model.evaluate(x, y)