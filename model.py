import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import tensorflow as tf
import os
import argparse
import re
import matplotlib.pyplot as plt 

with open('/home/daanvir/gg/project/SRM_project/x.csv') as f:
    shape = tuple(int(num) for num in re.findall(r'\d+', f.readline()))
X = np.loadtxt('/home/daanvir/gg/project/SRM_project/x.csv').reshape(shape)

Y = np.genfromtxt(
    r'/home/daanvir/gg/project/SRM_project/y.csv', delimiter=',')
yp=np.zeros((1024,8))
print(Y[0])

for i in range(1024):
    yp[i][int(Y[i])]=1
print(X,Y)
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(
    X, yp, test_size=0.2, random_state=415)

x_test = X

n_dim = 48


learning_rate = 0.01
training_epochs = 2000
batch_size = 100
display_step = 1000
model_path = r'/home/daanvir/gg/project/SRM_project/NMI'

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
x=[]
y=[]
#saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_epochs):
        #saver.restore(sess, model_path)
         # Apply softmax to logits 
        #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        _,c= sess.run([train_op,loss_op], feed_dict={X: x_test.reshape(1024, 48),Y: yp})
        
        
        
        #test_chkp = saver.save(sess, 'results/checkpoints/cp1.chkp')
        #tf.train.write_graph(sess.graph_def, 'results', 'model.pbtxt')
        if i%100==0:
            x.append(i)
            

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(yp, 1))
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print(c,i)
            print("Accuracy:", accuracy.eval({X: x_test, Y: yp}))
            y.append(accuracy.eval({X: x_test, Y: yp}))
            
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
plt.plot(x,y)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
        

            
 
#pred = tf.nn.softmax(logits) 
#print(pred)
    
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
