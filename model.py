import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
import os
import argparse

X = np.genfromtxt(
    r'/home/kartik/work/projects/motion/local/x.csv', delimiter=',')
Y = np.genfromtxt(
    r'/home/kartik/work/projects/motion/local/y.csv', delimiter=',')

X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.2, random_state=415)

x_test = np.genfromtxt(
    r'/home/kartik/work/projects/motion/local/x_pred.csv', delimiter=',')
n_dim = 48
n_class = 3
a = tf.Variable([0.3], tf.float32)
b = tf.Variable([0.4], tf.float32)
x = tf.placeholder(tf.float32, [None, n_dim])
y = tf.placeholder(tf.float32, [None, n_class])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))

learning_rate = 0.001
training_epochs = 5000
batch_size = 100
display_step = 1000
model_path = r'/home/kartik/work/projects/motion/local/NMI'

# Network Parameters
n_hidden_1 = 70  # 1st layer number of neurons
n_hidden_2 = 64  # 2nd layer number of neurons
n_hidden_3 = 32  # 3rd
n_hidden_4 = 16
n_input = n_dim  # MNIST data input (img shape: 28*28)
n_classes = 8  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),

    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    x = tf.identity(x, name="input_tensor")
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    player_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(player_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    player_2 = tf.nn.sigmoid(layer_2)
    layer_3 = tf.add(tf.matmul(player_2, weights['h3']), biases['b3'])
    player_3 = tf.nn.sigmoid(layer_3)
    layer_4 = tf.add(tf.matmul(player_3, weights['h4']), biases['b4'],)
    player_4 = tf.nn.sigmoid(layer_4)
    out_layer = tf.add(tf.matmul(player_4, weights['out']), biases['out'])
    return out_layer


# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    y_pred = tf.identity(pred, name="output_pred")
    y_pred_cls = tf.argmax(y_pred, axis=1)
    y_pred_cls = tf.identity(y_pred_cls, name="output_cls")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    pap = 0
    q = 0
    prediction_run = sess.run(pred, feed_dict={X: x_test.reshape(1, 48)})
    print(np.argmax(prediction_run))
    test_chkp = saver.save(sess, 'results/checkpoints/cp1.chkp')
    tf.train.write_graph(sess.graph_def, 'results', 'model.pbtxt')

# freeze_graph \
#   --input_graph=./model2/graph.pbtxt \
#   --input_checkpoint=./model2/model.ckpt-81852 \
#   --input_binary=false \
#   --output_graph=/tmp/frozen.pb \
#   --output_node_names=input_tensor,output_pred


#  ./bazel-bin/tensorflow/contrib/lite/toco/toco
#    --graph_def_file=/tmp/frozen.pb
#    --input_format=TENSORFLOW_GRAPHDEF
#    --output_format=TFLITE
#    --output_file=/tmp/cat_vs_dogs.tflite
#    --input_arrays=input_tensor
#    --output_arrays=output_pred
#    --input_shapes=1,224,224,3
