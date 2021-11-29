import cv2

#Reads image of a resistor
img = cv2.imread('test2.jpg', 1)

print(img)

# Code Below Scales down the image
scale_percent = 10  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

#Code displays image on screen
cv2.imshow('image', resized)
cv2.waitKey(0)