import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 6000 for training
train_x, train_y = mnist.train.next_batch(10000)

# 400 for testing
test_x, test_y = mnist.test.next_batch(400)

x_train = tf.placeholder("float", [None, 784])
x_test = tf.placeholder("float", [784])


# L1 distance

distance = tf.reduce_sum(tf.abs(tf.subtract(x_train, x_test)), reduction_indices=1)


# get min distance index
pred = tf.arg_min(distance, 0)

accuracy = 0.

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(len(test_x)):
        # get nearest neighbor
        nn_index = sess.run(pred, feed_dict={x_train: train_x, x_test: test_x[i, :]})
        print ("Test", i, "Prediction:", np.argmax(train_y[nn_index]), "True class:", np.argmax(test_y[i]))

        # calculate accuracy
        if np.argmax(train_y[nn_index]) == np.argmax(test_y[i]):
            accuracy += 1./len(test_x)

    print("Accuracy:", accuracy)





