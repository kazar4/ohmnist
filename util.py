import cv2
import numpy as np
from skimage import measure


def massCrop(img, tolerance, counter):

    # top to bottom
    y1 = 0
    y2 = img.shape[0]
    x1 = 0
    x2 = img.shape[1]

    oldMass = np.sum(img)/100
    while True:
        newMass = np.sum(img[counter:])/100
        #print(abs((newMass - oldMass)/oldMass))
        if abs((newMass - oldMass)/oldMass) < tolerance:
            img = img[counter:]
            y1 = y1 + counter
        else:
            break

    # bottom to top
    oldMass = np.sum(img)/100
    while True:
        newMass = np.sum(img[:img.shape[0] - counter])/100
        #print(abs((newMass - oldMass)/oldMass))
        if abs((newMass - oldMass)/oldMass) < tolerance:
            img = img[:img.shape[0] - counter]
            y2 = y2 - counter
        else:
            break

    # right to left
    oldMass = np.sum(img)/100
    while True:
        newMass = np.sum(img[:, counter:])/100
        #print(abs((newMass - oldMass)/oldMass))
        if abs((newMass - oldMass)/oldMass) < tolerance:
            img = img[:, counter:]
            x1 = x1 + counter
        else:
            break

    # left to right
    oldMass = np.sum(img)/100
    while True:
        newMass = np.sum(img[:, :img.shape[1] - counter])/100
        #print(abs((newMass - oldMass)/oldMass))
        if abs((newMass - oldMass)/oldMass) < tolerance:
            img = img[:, :img.shape[1] - counter]
            x2 = x2 - counter
        else:
            break
    
    return y1,y2,x1,x2


def cutOffWhite(img, mask):

    h = img.shape[0] * 255

    imgT = np.transpose(img, (1,0,2))
    maskT = mask.T

    newImg = []
    for i, col in enumerate(maskT):
        if np.sum(col) > h/2:
            newImg.append(imgT[i])
    
    #print(np.array(newImg).shape)

    return np.transpose(np.array(newImg), (1,0,2))


def splitImgToX(img, x):
    steper = img.shape[1] // x

    images = []

    for i in range(0,img.shape[1],steper):
        #print(i)
        images.append(img[:,i:i+steper,:])
    
    return np.array(images)

#https://rcvaram.medium.com/glare-removal-with-inpainting-opencv-python-95355aa2aa52
def create_mask(image):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    blurred = cv2.GaussianBlur( gray, (9,9), 0 )
    _,thresh_img = cv2.threshold( blurred, 180, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode( thresh_img, None, iterations=2 )
    thresh_img  = cv2.dilate( thresh_img, None, iterations=4 )
    # perform a connected component analysis on the thresholded image,
    # then initialize a mask to store only the "large" components
    labels = measure.label(thresh_img, background=0)
    mask = np.zeros( thresh_img.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique( labels ):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros( thresh_img.shape, dtype="uint8" )
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero( labelMask )
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add( mask, labelMask )
    return mask



# https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
def getDomColor(img):
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    try:
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    except:
        return None, False

    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    return dominant, True