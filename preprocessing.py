import cv2
import numpy as np
import math
import glob

def preprocess():
    filtered_images = []
    limit = 10
    count = 0
    for imgfile in glob.glob('./dtd/images/banded/*'):
        if(count == limit):
            break
        img = np.array(cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE))
        v = np.median(img)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (0.66) * v))
        upper = int(min(255, (0.66) * v))
        img = cv2.resize(img, (512, 512))
        img = np.array(cv2.Canny(img, lower, upper))
        filtered_images.append(img)
        count += 1
        
    count = 0
    for imgfile in glob.glob('./dtd/images/blotchy/*'):
        if(count == limit):
            break
        img = np.array(cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE))
        v = np.median(img)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (0.66) * v))
        upper = int(min(255, (0.66) * v))
        img = cv2.resize(img, (512, 512))
        img = np.array(cv2.Canny(img, lower, upper))
        filtered_images.append(img)
        count += 1
    return np.array(filtered_images)

preprocess()