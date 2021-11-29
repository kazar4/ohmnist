import cv2
import numpy as np
from util import *

#Image of resistor
frame = cv2.imread('test5.jpeg', 1)

hue = [29.136690647482013, 114.29541595925298]
sat = [107.77877697841727, 255.0]
val = [64.20863309352518, 255.0]

out = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
filtered = cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

c, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
thresh2Color = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
cv2.drawContours(thresh2Color, c, -1, (0, 255, 0), 2)

cv2.imshow("contours data", thresh2Color)
cv2.waitKey(0)

# calculate moments of binary image
M = cv2.moments(filtered)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv2.circle(thresh2Color, (cX, cY), 30, (255, 0, 0), 30)
cv2.putText(thresh2Color, "centroid", (cX - 200, cY - 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 10)

for cnt in c:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(thresh2Color,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("masked data", thresh2Color)
cv2.waitKey(0)

y1,y2,x1,x2 = massCrop(filtered, 0.10, 10)
print(y1)
print(y2)
print(x1)
print(x2)

mask_reverse = cv2.bitwise_not(filtered)
cv2.imshow("masked reverse data", mask_reverse)
cv2.waitKey(0)

masked_out = cv2.bitwise_and(frame, frame, mask=mask_reverse)

cv2.imshow("masked data", masked_out)
cv2.waitKey(0)

cropped_out = masked_out[y1:y2, x1:x2, :]
print(cropped_out.shape)

cv2.imshow("cropped data", cropped_out)
cv2.imshow("cropped data2", filtered[y1:y2, x1:x2])
cv2.imshow("cropped data3", mask_reverse[y1:y2, x1:x2])
cv2.waitKey(0)

squish_out = cutOffWhite(cropped_out, mask_reverse[y1:y2, x1:x2])
cv2.imshow("squish data", squish_out)
cv2.waitKey(0)

c, hierarchy = cv2.findContours(squish_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
thresh2Color = cv2.cvtColor(squish_out, cv2.COLOR_GRAY2RGB)
cv2.drawContours(thresh2Color, c, -1, (0, 255, 0), 2)

cv2.imshow("contours data", thresh2Color)
cv2.waitKey(0)


"""
parent = hierarchy[0, :, 3]

# Find parent contour with the maximum number of child contours
# Use np.bincount for counting the number of instances of each parent value.
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html#numpy.bincount
hist = np.bincount(np.maximum(parent, 0))
max_n_childs_idx = hist.argmax()

# Get the contour with the maximum child contours
c = c[max_n_childs_idx]

# Get bounding rectangle
x, y, w, h = cv2.boundingRect(c)
print(x)
print(y)
print(w)
print(h)

# Crop the bounding rectangle out of img
thresh2Color = thresh2Color[y:y+h, x:x+w, :]
gray = cv2.cvtColor(thresh2Color, cv2.COLOR_BGR2GRAY)

cv2.imshow("masked data", gray)
cv2.waitKey(0)
"""

"""
BoundingBoxX = int(frame.shape[0] / 4)
BoundingBoxY = int(frame.shape[1] / 4)
BoundingBoxW = int(frame.shape[1] / 4 + frame.shape[1] / 2)
BoundingBoxH = int(frame.shape[0] / 4 + frame.shape[0] / 2)

#Scales image down to 30% of size
scale_percent = 10  # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Median Blur
median = cv2.medianBlur(frame, 23)

#Converts image to gray scale
img = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

#Applys threshold
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 49, 5)

cv2.imshow("Mean Adaptive Thresholding", thresh2)
cv2.waitKey(0)

mask = cv2.bitwise_not(thresh2)
masked_data = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow("masked data", masked_data)
cv2.waitKey(0)

#287 y
#383 x
#Inverts Threshold
thresh2 = cv2.bitwise_not(thresh2)

#Find contours, and displays contours as green
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
thresh2Color = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2RGB)
cv2.drawContours(thresh2Color, contours, -1, (0, 255, 0), 2)

#Creates bounding box for anaylsis
cv2.rectangle(thresh2Color, (BoundingBoxX, BoundingBoxY), (BoundingBoxW, BoundingBoxH), (0, 0, 255), 2)

#Displays video with contours
cv2.imshow('video', thresh2Color)
"""

#capture.release()
cv2.destroyAllWindows()
