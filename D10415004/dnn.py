from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn import cross_validation
import numpy as np


np.set_printoptions(suppress=True)
##############
#filepath=sys.argv[1]
filepath = "/Users/scml/KDD2017/kdd/trainraw.csv.7v2"
f = open(filepath)
dataset = np.loadtxt(f,delimiter=',',skiprows=1)
Y_train=dataset[:,0]
X_train=dataset[:,1:]
print(Y_train[0])
print(X_train[0])
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)

total_len = X_train.shape[0]

filepath = "/Users/scml/KDD2017/kdd/testraw.csv.7v2"
f = open(filepath)
dataset2 = np.loadtxt(f,delimiter=',',skiprows=1)
Y_test=dataset2[:,0]
X_test=dataset2[:,1:]

X_test = scaler.fit_transform(X_test)

# Parameters
learning_rate = 0.001
training_epochs = 10000
batch_size = 50
display_step = 1
dropout_rate = 0.9
# Network Parameters
n_hidden_1 = 32 # 1st layer number of features
n_hidden_2 = 200 # 2nd layer number of features
n_hidden_3 = 200
n_hidden_4 = 256
n_input = X_train.shape[1]
n_classes = 1

# tf Graph input
x = tf.placeholder("float", [None, 15])
y = tf.placeholder("float", [None])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
pred = tf.transpose(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            #print ("batch_x")
            #print (batch_x)
            #print ("batch_y")
            #print (batch_y)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # sample prediction
        label_value = batch_y
        estimate = p
        err = label_value-estimate
        print ("num batch:", total_batch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            #for i in xrange(3):
            #    print ("label value:", label_value[i], \
            #        "estimated value:", estimate[i])
            #print ("[*]============================")

    print ("Optimization Finished!")

    scaler = preprocessing.MinMaxScaler()
    X_test = scaler.fit_transform(X_test)
    result = []
    for i in X_test:
        i = [i]
        predValue = sess.run(pred, feed_dict={x:i})
        result.append(predValue[0][0])
    
        
i=0
f=open('/Users/scml/KDD2017/kdd/testuniqid', "rb")
print('intersection_id,tollgate_id,time_window,avg_travel_time')
for line in f :
  tmp=line.split(';')
  print(tmp[0]+','+tmp[1]+','+tmp[2].strip()+','+str(round(result[i],2)))
  i=i+1
    