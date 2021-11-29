import cv2
import numpy as np


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
    
    return np.transpose(np.array(newImg), (1,0,2))

