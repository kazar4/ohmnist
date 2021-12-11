import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

#capture = cv2.VideoCapture(0)

while (True):

    #Image of resistor
    #Image of resistor
    frame = cv2.imread('test5.jpeg', 1)

    BoundingBoxX = int(frame.shape[0] / 4)
    BoundingBoxY = int(frame.shape[1] / 4)
    BoundingBoxW = int(frame.shape[1] / 4 + frame.shape[1] / 2)
    BoundingBoxH = int(frame.shape[0] / 4 + frame.shape[0] / 2)

    #Video from webcam
   # ret, frame = capture.read()

    #Scales image down to 30% of size
    scale_percent = 10  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    #resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    #resized = image_resize(frame, height=300)

    # Median Blur
    median = cv2.medianBlur(frame, 23)

    #Converts image to gray scale
    img = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

    #Applys threshold
    #thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 49, 15)

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

    #When "esc" key is pressed the current frame will be saved
    if cv2.waitKey(1) == 27:
        break

"""

frame2 = frame
c = contours
captureColor = frame
captureThresh = thresh2
captureThreshColor = thresh2Color

#Crops image to boudning box
captureColor = captureColor[BoundingBoxY:BoundingBoxH, BoundingBoxX:BoundingBoxW]
captureThresh = captureThresh[BoundingBoxY:BoundingBoxH, BoundingBoxX:BoundingBoxW]
captureThreshColor = captureThreshColor[BoundingBoxY:BoundingBoxH, BoundingBoxX:BoundingBoxW]

#cv2.imshow("Image", captureThreshColor)
#cv2.waitKey(0)

# calculate moments of binary image
M = cv2.moments(captureThresh)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv2.circle(captureColor, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(captureColor, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#cv2.imshow("centroid", captureColor)
#cv2.waitKey(0)

#Puts a bounding box for each contour
for cnt in c:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(captureColor,(x,y),(x+w,y+h),(0,255,0),2)


masked_data = cv2.bitwise_and(captureColor, captureColor, mask=captureThresh)
cv2.imshow("masked data", masked_data)
cv2.waitKey(0)

frame_HLS = cv2.cvtColor(masked_data, cv2.COLOR_BGR2HLS)
frame_threshold = cv2.inRange(frame_HLS, (50, 0, 0), (139, 149, 255))

#frame_threshold = cv2.adaptiveThreshold(masked_data, 255,
#	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2)
#t = cv2.bitwise_and(img, frame_threshold)
#t = cv2.bitwise_and(captureColor, captureColor, mask=frame_threshold)
#cv2.imshow("Mean Adaptive Thresholding", frame_threshold)
#cv2.waitKey(0)

frame_threshold = cv2.bitwise_not(frame_threshold)

# display the image with centroid and bounding box

scale_percent = 1000  # percent of original size
#width = int(frame_threshold.shape[1] * scale_percent / 100)
#height = int(frame_threshold.shape[0] * scale_percent / 100)
width = 800
height = 400
dim = (width, height)
masked_data3 = cv2.resize(frame_threshold, dim, interpolation=cv2.INTER_AREA)

frame_threshold = image_resize(frame_threshold, height=3024)

#masked_data2 = cv2.bitwise_and(masked_data, masked_data, mask=frame_threshold)

cv2.imshow("Image", masked_data3)
cv2.waitKey(0)

"""

#capture.release()
cv2.destroyAllWindows()