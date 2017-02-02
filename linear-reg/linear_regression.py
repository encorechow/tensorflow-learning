import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ran = np.random

lr = 0.001
epochs = 1000
display_step = 50


train_x = np.array([ran.random_sample() for _ in range(50)])
train_y = np.array([ran.random_sample() for _ in range(50)])


n_samples = train_x.shape[0]


x = tf.placeholder("float")
y = tf.placeholder("float")

w = tf.Variable(ran.randn(), name="weight")
b = tf.Variable(ran.randn(), name="bias")


pred = tf.add(tf.mul(x, w), b)


cost = tf.reduce_sum(tf.pow(pred-y, 2))/n_samples

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for (xt, yt) in zip (train_x, train_y):
            sess.run(optimizer, feed_dict={x: xt, y: yt})

        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={x: xt, y: yt})
            print('Epoch', '%04d' % (epoch+1), 'cost=', '{:.9f}' .format(c), 'w=', sess.run(w), 'b=', sess.run(b))

    training_cost = sess.run(cost, feed_dict={x:train_x, y: train_y})
    print('training cost=', training_cost, 'w=', sess.run(w), 'b=', sess.run(b))


    plt.plot(train_x, train_y, 'ro', label='original data')
    plt.plot(train_x, sess.run(w)*train_x + sess.run(b), label='fitted line')
    plt.legend()
    plt.show()

