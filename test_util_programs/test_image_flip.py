import cv2
import tensorflow as tf
import numpy as np
from preprocess import get_data
#pip install opencv-python

data = get_data("/Volumes/POGDRIVE/trainR.db", "/Volumes/POGDRIVE/testR.db")[0]
counter = 0
for pair in data:
    frame = np.array(pair[0])
    label = np.array(pair[1])
    if counter == 0:
        print(label)
        break
    counter += 1


# give an image
# frame = np.array(cv2.imread("/Users/joshabramson1/Desktop/celeste_screenshot.png"))
print(frame.shape)
cv2.imshow("unflipped", frame)
cv2.waitKey(0)

flipped_image = np.array(tf.reverse(frame, [1])) # FLIP HERE
flipped_label = np.array(tf.reverse(label, [0]))
print(flipped_label)
print(frame.shape)
cv2.imshow("flipped", flipped_image)
cv2.waitKey(0)

