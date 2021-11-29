from preprocess_pipe import *
import os
import cv2

#dir = "../Downloads"
dir = "./test_folder"
for i, files in enumerate(os.listdir(dir)):
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
    
        cv2.imwrite(str(save_path), pair[1])
    
    print(f"image {i} finished")