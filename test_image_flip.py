import cv2
import tensorflow as tf
import numpy as np
from preprocess import get_data
#pip install opencv-python

data = get_data("/Volumes/POGDRIVE/trainR.db", "/Volumes/POGDRIVE/testR.db")[0]
counter = 0
for pair in data:
    frame = np.array(pair[0])
    if counter == 0:
        print(frame)
        print(pair[1])
        break
    counter += 1


# give an image
# frame = np.array(cv2.imread("/Users/joshabramson1/Desktop/celeste_screenshot.png"))
print(frame.shape)
cv2.imshow("unflipped", frame)
cv2.waitKey(0)

flipped_image = tf.reverse(frame, [1]) # FLIP HERE
print(frame.shape)
cv2.imshow("flipped", flipped_image)
cv2.waitKey(0)

