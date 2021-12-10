import cv2
#pip install opencv-python

# give an image
frame = cv2.imread("some image", 1)

cv2.imshow("unflipped", frame)
cv2.waitKey(0)

flipped_image = frame # FLIP HERE

cv2.imshow("flipped", flipped_image)
cv2.waitKey(0)

