import cv2
import numpy as np
import math
import glob

def preprocess():
    target = []
    filtered_images = []
    limit = 5
    classifier_count = 0
    classifier_total = 0
    # For each folder
    for img_type in glob.glob('./dtd/images/*/'):
        classifier_total += 1
    for img_type in glob.glob('./dtd/images/*/'):
        print(img_type)
        count = 0
        classifier = np.zeros(classifier_total)
        classifier[classifier_count] = 1
        # For each image
        for imgfile in glob.glob(img_type + "*"):
            if(count == limit):
                break
            img = np.array(cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE))
            v = np.median(img)
            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (0.66) * v))
            upper = int(min(255, (0.66) * v))
            img = cv2.resize(img, (512, 512))
            img = np.array(cv2.Canny(img, lower, upper))
            img = np.divide(img, 255)
            img = img.flatten()
            filtered_images.append(img)
            count += 1
            target.append(classifier)
        classifier_count += 1
    return np.array(filtered_images), np.array(target), classifier_count