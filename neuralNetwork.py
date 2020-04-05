import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import warnings
import glob
warnings.filterwarnings('ignore')


numberOfTextures = 2#47
maxNum = 10
imagearr = []
for folder in glob.glob("dtd/images/*"):
    count =0
    for imageFile in glob.glob(folder+ "/*"):
        if count >= maxNum:
            break
        count +=1
        # print(imageFile)
        imagearr.append(cv2.imread(imageFile, flags=cv2.IMREAD_GRAYSCALE))



image = cv2.imread("dtd/images/banded/banded_0004.jpg", flags=cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image,(256,256))
# print(image)
# plt.imshow(image, cmap='gray')
# plt.show()

# print(image.shape)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(256*256,)))
model.add(tf.keras.layers.Dense(64, activation='softmax'))
model.add(tf.keras.layers.Dense(numberOfTextures))
model.compile(optimizer="adam", loss="categorical_crossentropy")