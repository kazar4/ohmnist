from preprocess_pipe import *
import os
import cv2
import tensorflow as tf
import numpy as np

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
    "Black":0,
    "Brown":1,
    "Red":2,
    "Orange":3,
    "Yellow":4,
    "Green":5,
    "Blue":6,
    "Violet":7,
    "Grey":8,
    "White":9,
    "Gold":10,
    "Silver":11
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

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))

    #print(dataset)
    tf.data.experimental.save(dataset, "./image_outputs/test.db")
    new_dataset = tf.data.experimental.load("./image_outputs/test.db")

    #for d in new_dataset:
    #    print(d) 
    #    print("_______________________________________________")
    
    print(f"image {i} finished")