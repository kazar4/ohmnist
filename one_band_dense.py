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

        # hyperparameters

        self.num_classes = num_classes
        self.batch_size = 25
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(num_classes * num_classes)
        self.dense2 = tf.keras.layers.Dense(num_classes)


    def call(self, inputs):
        """
        take in inputs, output probabilities for each class
        """
        flatten = self.flatten(inputs)

        l1out = self.dense1(flatten)
        logits = self.dense2(l1out)

        probs = tf.nn.softmax(logits)

        return probs
        
    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: calculated probabilities for each class
        :param labels: true class labels for each input
        :return: the loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

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
    losslist = []
    i = 0
    end = int(train_inputs.shape[0])
    while (i + model.batch_size) < end:
        with tf.GradientTape() as g:
            probs = model.call(train_inputs[i:i+model.batch_size])
            loss = model.loss(probs, train_labels[i:i+model.batch_size])
            losslist.append(loss)


        gradients = g.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
        i += model.batch_size
    
    if not(i == end):
        with tf.GradientTape() as g:
            probs = model.call(train_inputs[i:end])
            loss = model.loss(probs, train_labels[i:end])
            losslist.append(loss)

        gradients = g.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    losses_m1 = tf.data.Dataset.from_tensor_slices(losslist)
    tf.data.experimental.save(losses_m1, "./loss_densemodel.db")

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
    #data = get_data("./processed_data/train.db", "./processed_data/test.db")
    #data = get_data("/Volumes/POGDRIVE/train.db", "/Volumes/POGDRIVE/test.db")
    #data = get_data("F:/train.db", "F:/test.db")
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
    #print(test(model, train_inputs, train_labels))

    pass

if __name__ == '__main__':
    main()
