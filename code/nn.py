import numpy as np
import tensorflow as tf
#load data break into chunks to fit memory size and reshape the data to fit rnn input
charcters = np.load("../processed data/charcters.npy")
no_features = np.load("../processed data/no_features.npy")
no_features=no_features[0:int(len(no_features)/10)]
no_label = np.load("../processed data/no_label.npy")
no_label=no_label[0:int(len(no_label)/10)]
word_features = np.load("../processed data/word_features.npy")
word_label = np.load("processed data/word_label.npy")
print(no_features.shape)
print(no_label.shape)
print(word_features.shape)
print(word_label.shape)
x1, x2, x3, x4 = no_features[:, 0], no_features[:, 1], no_features[:, 2], no_features[:,
                                                                          3]  ## x are input to each layer of rnn
x1 = x1.reshape(x1.shape[0], 1)
x2 = x2.reshape(x2.shape[0], 1)
x3 = x3.reshape(x3.shape[0], 1)
x4 = x4.reshape(x4.shape[0], 1)
print(x1.shape)
## network parameters
no_of_neurons_layers = 20  ## all layers same neuron
batch_size = 128
display_step = 100
learning_rate = 0.01
num_steps = 500
# graph input
X1 = tf.placeholder("float", [None, 1])
X2 = tf.placeholder("float", [None, 1])
X3 = tf.placeholder("float", [None, 1])
X4 = tf.placeholder("float", [None, 1])
Y = tf.placeholder("float", [None, 16252])
# Store layers weight & bias
weights = {
    'waa1': tf.Variable(tf.random_normal([1])),
    'wax1': tf.Variable(tf.random_normal([1])),
    'wya': tf.Variable(tf.random_normal([1,16252])),
}
biases = {
    'ba': tf.Variable(tf.random_normal([1])),
    'by': tf.Variable(tf.random_normal([1]))
}
a0 = tf.Variable(tf.random_normal([1]))


def rnn(h1, h2, h3, h4):
    """

    :param h1: x1
    :param h2: x2
    :param h3: x3
    :param h4: x3
    :return: y sofmax output
    """
    a1 = tf.add(tf.multiply(a0, weights['waa1']), tf.multiply(h1, weights['wax1']))
    a1 = tf.add(a1, biases['ba'])
    a1 = tf.nn.relu(a1)
    a2 = tf.add(tf.multiply(a1, weights['waa1']), tf.multiply(h2, weights['wax1']))
    a2 = tf.add(a2, biases['ba'])
    a2 = tf.nn.relu(a2)
    a3 = tf.add(tf.multiply(a2, weights['waa1']), tf.multiply(h3, weights['wax1']))
    a3 = tf.add(a3, biases['ba'])
    a3 = tf.nn.relu(a3)
    a4 = tf.add(tf.multiply(a3, weights['waa1']), tf.multiply(h4, weights['wax1']))
    a4 = tf.add(a4, biases['ba'])
    a4 = tf.nn.relu(a4)
    y = tf.nn.relu(tf.add(tf.multiply(weights['wya'], a4), biases['by']))
    return y


# Construct model
logits = rnn(X1, X2, X3, X4)
prediction = tf.nn.softmax(logits)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps + 1):
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X1: x1, X2: x2, X3: x3, X4: x4, Y: no_label})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X1: x1, X2: x2, X3: x3, X4: x4, Y: no_label})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
