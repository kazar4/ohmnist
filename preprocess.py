
import tensorflow as tf

def get_data(train_dir, test_dir):

    train_data = tf.data.experimental.load(train_dir)
    test_data = tf.data.experimental.load(test_dir)

    return train_data, test_data