from preprocess_pipe import *
import os
import cv2
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#dir = "../Downloads"
#dir = "./data/workingPhotos"
#dir = "./data/DATASET/data"
dir = "./data/data/"
directory_files = os.listdir(dir)
total_images = len(directory_files)
tensor_dim = (total_images*5, 200, 60, 3)
#200, 60, 3)
data = np.zeros(tensor_dim)
labels = np.zeros(total_images*5)

dataR = np.zeros((total_images, 300, 600, 3))
labelsR = np.zeros((total_images, 5))

#https://www.digikey.com/en/articles/big-boys-race-young-girls-violet-wins-resistors-color-codes
# Color chart being used for dictionary, going in order from top to bottom
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

newList = []
for i in directory_files:
    if i[-4:].lower() == ".png" or i[-4:].lower() == ".jpg" or i[-5:].lower() == ".jpeg":
        newList.append(i)

# https://thispointer.com/python-get-list-of-files-in-directory-sorted-by-date-and-time/
newList = sorted(newList,
                        key = lambda x: os.path.getctime(os.path.join(dir, x)))

for i, files in enumerate(newList):
    print(files)
    
    #print(str(i), files)
    #print(str(i), os.path.join(dir, files))

    # gets image path and creates processed images
    img_path = os.path.join(dir, files)
    try:
        generated_images, entire_resistor = pipeline(img_path)
    except:
        print(f"Error with image: {files}")
        print("Skipped for now")
        continue

    # create outout folder if it doesnt exist
    if not os.path.isdir("image_outputs"):
        os.mkdir("image_outputs")

    # edit file name string
    files = files.lower()
    files = files.replace(".png", "")
    files = files.replace(".jpg", "")
    files = files.replace(".jpeg", "")

    # create resistor folder if it doesnt exist
    if not os.path.isdir(os.path.join("image_outputs", files)):
        os.mkdir(os.path.join("image_outputs", files))

    # saves file as label name
    labelToFile = zip(files.split(" ")[1:], generated_images)
    for j, pair in enumerate(labelToFile):

        dirPath = str(os.path.join("image_outputs", files))
        save_path = os.path.join(dirPath, f"{i} {j} {pair[0]}.png")
        # print(str(save_path))

        data[i*5 + j] = pair[1]
        labels[i*5 + j] = colorsToNum[pair[0]]

        cv2.imwrite(str(save_path), pair[1])

        #test = np.vstack([data, labels])
        #test = list(zip(data, labels))
        #print(test)
    
    dataR[i] = entire_resistor
    labelsR[i] = np.array([colorsToNum[cTxt] for cTxt in files.split(" ")[1:]])

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=42)

    X_trainR, X_testR, y_trainR, y_testR = train_test_split(dataR, labelsR, test_size=0.15, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_datasetR = tf.data.Dataset.from_tensor_slices((X_trainR, y_trainR))
    test_datasetR = tf.data.Dataset.from_tensor_slices((X_testR, y_testR))

    #print(dataset)
    tf.data.experimental.save(train_dataset, "./image_outputs/train.db")
    tf.data.experimental.save(test_dataset, "./image_outputs/test.db")

    tf.data.experimental.save(train_datasetR, "./image_outputs/trainR.db")
    tf.data.experimental.save(test_datasetR, "./image_outputs/testR.db")

    new_dataset = tf.data.experimental.load("./image_outputs/test.db")

    #for d in new_dataset:
    #    print(d) 
    #    print("_______________________________________________")
    
    print(f"image {i} finished")