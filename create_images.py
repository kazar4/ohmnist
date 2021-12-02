from preprocess_pipe import *
import os
import cv2
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#dir = "../Downloads"
dir = "./test_database"
directory_files = os.listdir(dir)
total_images = len(directory_files)
tensor_dim = (total_images*5, 200, 60, 3)
#200, 60, 3)
data = np.zeros(tensor_dim)
labels = np.zeros(total_images*5)

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
    "violet":7,
    "grey":8,
    "white":9,
    "gold":10,
    "silver":11
}

for i, files in enumerate(directory_files):
    #print(str(i), files)
    #print(str(i), os.path.join(dir, files))

    # gets image path and creates processed images
    img_path = os.path.join(dir, files)
    generated_images = pipeline(img_path)

    # create outout folder if it doesnt exist
    if not os.path.isdir("image_outputs"):
        os.mkdir("image_outputs")

    # edit file name string
    files = files.lower()
    files = files.replace(".png", "")
    files = files.replace(".jpg", "")
    files = files.replace(".jpeg", "")

    # saves file as label name
    labelToFile = zip(files.split(" "), generated_images)
    for j, pair in enumerate(labelToFile):
        save_path = os.path.join("image_outputs", f"{i} {j} {pair[0]}.png")

        data[i*5 + j] = pair[1]
        labels[i*5 + j] = colorsToNum[pair[0]]

        cv2.imwrite(str(save_path), pair[1])

        #test = np.vstack([data, labels])
        #test = list(zip(data, labels))
        #print(test)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    #print(dataset)
    tf.data.experimental.save(train_dataset, "./image_outputs/test.db")
    tf.data.experimental.save(test_dataset, "./image_outputs/test.db")

    new_dataset = tf.data.experimental.load("./image_outputs/test.db")

    #for d in new_dataset:
    #    print(d) 
    #    print("_______________________________________________")
    
    print(f"image {i} finished")