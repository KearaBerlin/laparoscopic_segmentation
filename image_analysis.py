import cv2
import numpy as np


import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    return image

image = cv2.imread('images/hliv_1.jpg')
ref_image = cv2.imread('images/hliv_2.jfif')

print(load_image(image)) # error check
print(load_image(ref_image)) #error check

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
_, ref_thresh = cv2.threshold(ref_gray, 127, 255, cv2.THRESH_BINARY_INV)
ref_contours, _ = cv2.findContours(ref_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ref_contour = max(ref_contours, key=cv2.contourArea)

# will check for any match, if true, similarity score will get a hit. Current threshold that always works (0.3)
for contour in contours:
    similarity = cv2.matchShapes(ref_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
    print(f"Similarity: {similarity}")

    if similarity < 0.05:  # example threshold
        print("Found a similar shape!")
