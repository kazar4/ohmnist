import numpy as np

colorsToNum = {
    "black":0,
    "brown":1,
    "red":2,
    "orange":3,
    "yellow":4,
    "green":5,
    "blue":6,
    "purple":7,
    "violet":7,
    "grey":8,
    "gray":8,
    "white":9,
    "gold":10,
    "silver":11
}

"""
def colorsAcc(probs, labels):
    print(labels.shape)

    acc = np.zeros(12)
    counts = np.zeros(12)

    for i in range(0,12):
        colorP = []
        colorL = []
        
        colorC = 0

        for j, val in enumerate(labels):
            if val == i:
                colorL.append(labels[j])
                colorP.append(probs[j])
                colorC = colorC + 1

        colorL = np.array(colorL)
        colorP = np.array(colorP)
        print(colorL.shape)
        print(colorP.shape)

        acc[i] = np.mean(np.array(colorL) == np.argmax(np.array(colorP), axis = 1))
        counts[i] = colorC
    
    return acc, counts
"""

def colorsAcc(probs, labels):

    acc = np.zeros(12)
    counts = np.zeros(12)
    
    probs = probs.numpy()
    
    for i in range(0,12):
        indices = np.nonzero(labels == i)[0]
        indices = list((indices.astype(int).T))

        counts[i] = len(indices)

        colorL = labels[indices]
        colorP = probs[indices]

        #print(indices)
        #print("nL", colorL.shape)
        #print("nP", colorP.shape)

        acc[i] = np.mean(colorL == np.argmax(colorP, axis = 1))
    
    return acc, counts