import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.python.ops.gen_array_ops import empty
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, num_classes):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        #hyperparameters

        self.num_classes = num_classes
        self.batch_size = 25
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        self.image_width = 200
        self.image_height = 60
        self.input_channels = 3

        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), padding="valid", activation="relu", data_format="channels_last", input_shape=(self.image_width, self.image_height, self.input_channels))
        self.conv2 = tf.keras.layers.Conv2D(16, (5, 5), padding="valid", activation="relu", data_format="channels_last")
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1,1), padding="valid")
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding="valid")

        self.dense1 = tf.keras.layers.Dense(5 * num_classes)


    def call(self, inputs):
        """
        take in inputs, output probabilities for each class
        """

        c1out = self.conv1(inputs)
        p1out = self.pool1(c1out)
        c2out = self.conv2(p1out)
        p2out = self.pool2(c2out)
        logits = self.dense1(p2out)

        probs = tf.nn.softmax(logits)

        return probs
        
    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: calculated probabilities for each class
        :param labels: true class labels for each input
        :return: the loss of the model as a tensor of size 1
        """

        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, probs))

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # calculate the batch accuracy
        return np.mean(labels == np.argmax(probabilities, axis = 1))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    shuffle = np.arange(len(train_labels))			#reorder for mixed batches
    np.random.shuffle(shuffle)
    tf.gather(train_inputs, shuffle)
    tf.gather(train_labels, shuffle)

    i = 0
    end = int(train_inputs.shape[0])
    while (i + model.batch_size) < end:
        with tf.GradientTape() as g:
            logits = model.call(train_inputs[i:i+model.batch_size])
            loss = model.loss(logits, train_labels[i:i+model.batch_size])
            model.loss_list.append(loss)

        gradients = g.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
        i += model.batch_size
    
    if not(i == end):
        with tf.GradientTape() as g:
            logits = model.call(train_inputs[i:end])
            loss = model.loss(logits, train_labels[i:end])
            model.loss_list.append(loss)

        gradients = g.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: list of losses
    """

    probs = model.call(test_inputs)
    return model.accuracy(probs, test_labels)


def main():
    # Pre-process and vectorize the data
    # data = get_data("./process_data/train.db", "./process_data/test.db")
    data = get_data("/Volumes/POGDRIVE/train.db", "/Volumes/POGDRIVE/test.db")
    train_data = data[0]
    test_data = data[1]
    # initialize model
    model = Model(12) #12 classes of colors to identify

    #train data needs to be set here TODO
    
    train_inputs = np.array([pair[0] for pair in train_data])
    train_labels = np.array([pair[1] for pair in train_data])
    train(model, train_inputs, train_labels)

    #need to do test data TODO
    test_inputs = np.array([pair[0] for pair in test_data])
    test_labels = np.array([pair[1] for pair in test_data])
    # Print out loss 
    print(test(model, test_inputs, test_labels))

    pass

if __name__ == '__main__':
    main()
