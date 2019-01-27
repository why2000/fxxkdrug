#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:38:23 2019

@author: Wu Hanyuan
"""

import tensorflow as tf

def read_data():
    pass


data = read_data()

# Training Parameters
num_cities = 460
num_years = 6
learning_rate = 0.001
num_steps = 5
stddev = 1.0



with tf.name_scope('Placeholders'):
    V = tf.placeholder(tf.float32, [num_cities, num_years], name='infected')
    U = tf.placeholder(tf.float32, [num_cities, num_years], name='infect_rate')

# Set model weights
with tf.name_scope('Weights'):
    weights = {
        'G': tf.Variable(tf.random_normal([num_cities,num_cities,num_years])),
    }
    
    biases = {
        'e': tf.Variable(tf.random_normal([436], mean=0.0, stddev=stddev))
    }

def forward(X, weights):
    X = V
    out = tf.matmul(G,X[:,-2])
    return out
with tf.name_scope('Model'):
    y_pred = 


with tf.name_scope('Loss'):
    loss = 0
    for i in range(num_years-1):
        loss += tf.norm(U[:,i]-G*V[:,i], ord=2, axis=None)**2
    loss = loss/(stddev**2)
    

with tf.name_scope('Train_ops'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
with tf.name_scope('Metrics'):
    Y_true = V[-1]

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    merged=tf.summary.merge_all()
    
with tf.name_scope('Misc'):
    init = tf.global_variables_initializer()
    
train_writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:

    sess.run(init)
    for step in range(1, num_steps+1):
        batch_x, batch_y = next_batch(batch_size)
        feed_dictionary={X: batch_x, Y: batch_y, keep_prob: dropout}
        # Run optimization op (backprop)
        _, summary, t_loss = sess.run([train_op,merged,loss], feed_dict=feed_dictionary)
        train_writer.add_summary(summary, step)
#
#        
    print("Optimization Finished!")
    

   


sess.close()
train_writer.close()


