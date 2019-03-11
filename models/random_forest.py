import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
import matplotlib.pyplot as plt 
import numpy as np
# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

with open('/home/daanvir/gg/project/SRM_project/x.csv') as f:
    shape = tuple(int(num) for num in re.findall(r'\d+', f.readline()))
X = np.loadtxt('/home/daanvir/gg/project/SRM_project/x.csv').reshape(1024,48)

Y = np.genfromtxt(
    r'/home/daanvir/gg/project/SRM_project/y.csv', delimiter=',')
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.2, random_state=415)

x=[]
y=[]
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Parameters
num_steps = 50 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = 8 # The 10 digits
num_features = 48# Each image is 28x28 pixels
num_trees = 10
max_nodes = 1000

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

saver=tf.train.Saver()

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    
    _, l = sess.run([train_op, loss_op], feed_dict={X: train_x, Y: train_y})
    if i % 2 == 0 or i == 1:
        x.append(i)
        acc = sess.run(accuracy_op, feed_dict={X: train_x, Y: train_y})
        y.append(acc)
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
plt.plot(x,y)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
save_path=saver.save(sess,"/home/daanvir/gg/project/SRM_project/motion_ml/results/model.ckpt")
# Test Model

print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))