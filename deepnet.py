import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
input > weighted >hidden layer 1 (activation function) > weights
> hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost / loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropagation


feed forward + backprop = epoch

'''

'''
10 classes 0-9 numbers
one_hot examples:
    0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500


n_classes = 10

batch_size = 100

# height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])


def neural_network_model(data):

    # (input_data * weights) + biases

    hidden_1_layer = {
                        'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                        'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
                     }

    hidden_2_layer = {
                        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
                     }

    hidden_3_layer = {
                        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
                     }

    output_layer = {
                        'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases': tf.Variable(tf.random_normal([n_classes]))
                     }


    # layer 1
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)


    # layer 2
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    # layer 3
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    # output
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # default learning rate = 0.001 for AdamOptimi zer
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    # cycles feed forward + backprop
    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())


        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of ', epochs, ' loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
