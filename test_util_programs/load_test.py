import tensorflow as tf

new_dataset = tf.data.experimental.load("/Volumes/POGDRIVE/train.db")
#new_dataset.batch(3, drop_remainder=True)

for i, val in enumerate(new_dataset):
    print(f"{i}, {val[0].shape}   :   {val[1].shape} ")

#for val in new_dataset:
#    (image, label)
inputs = [pair[0] for pair in new_dataset]
labels = [pair[1] for pair in new_dataset]

#print(new_dataset.shape)