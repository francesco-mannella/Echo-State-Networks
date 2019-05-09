#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from ESN import EchoStateRNNCell, MinMax

import keras
from keras.layers import Input, Layer
from keras.models import Sequential
from keras.regularizers import l2    

# Configs ----------------------------------------------------------------------

# takes only current needed GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# random numbers
random_seed = np.random.randint(1, 1e5)
rng = np.random.RandomState(random_seed)

# Utils ------------------------------------------------------------------------

def mackey_glass(stime = 30000, dt = 0.05,
        beta = 2., gamma = 1., n = 9.65, tau = 2):
    d = int(tau/dt)    
    x = 3.0*np.ones(2*stime + d)
    for t in range(tau-1, 2*stime):
        x[t+1] = x[t] + dt*(
                beta*x[t-d] / (1.0 + x[t-d]**n)
                - gamma*x[t] )
    return x[2*d: stime + 2*d], d

def mult_sines(stime = 1200):
    res = np.arange(stime)
    res = np.sin(res) \
            + np.sin(0.51*res) \
            + np.sin(0.22*res) \
            + np.sin(0.1002*res) \
            + np.sin(0.05343*res)
    
    res -= res.min() 
    res /= res.max()
    
    return res
   
class RidgeRegression(keras.layers.Layer):

    def __init__(self, targets, num_params, lmb, **kargs):
        self.num_params = num_params
        self.targets = tf.constant(targets)
        self.output_dim = targets.shape[1:]
        self.lmb_init = lmb
        super(RidgeRegression,  self).__init__(**kargs)
    
    def build(self, input_shape):
        self.lmb = self.add_weight(name='lambda', 
                shape=(), initializer = keras.initializers.Constant(self.lmb_init), 
                constraint = MinMax(0.00001, 0.1), 
                trainable = True)
        super(RidgeRegression, self).build(input_shape)
    
    def regression(self, elements):
        inp, target = elements
        os_weights = tf.matmul( tf.matrix_inverse(
            tf.matmul(tf.transpose(inp), inp) + self.lmb*tf.eye(self.num_params)),
            tf.matmul(tf.transpose(inp), target))
        out = tf.matmul(inp, os_weights)
        return out 

    def call(self, inputs):
        return tf.map_fn(self.regression, [inputs, self.targets], dtype=tf.float32)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.output_dim

# variables --------------------------------------------------------------

batches = 1
stime = 700
num_units = 20
num_inputs = 1
num_outputs = 1
lr = 0.00001
lmb_init = 0.001
timewindow_begin = 50
timewindow_end = stime
regression_time = timewindow_end - timewindow_begin

# the activation function of the ESN
out_function = lambda x:  math_ops.tanh(x)


# input -------------------------------------------------------------------------
wave = mult_sines(stime+5).astype("float32")
rnn_inputs = wave[5:].reshape(batches, stime, num_inputs).astype("float32")

# the output target - mackey-glass ----------------------------------------------
rnn_target = wave[:-5]
rnn_target = rnn_target.reshape(batches, stime, num_outputs).astype("float32")

# tensorflow graph -------------------------------------------------------------
cell = EchoStateRNNCell(num_units=num_units, 
        activation=out_function, 
        decay=0.15, 
        alpha=0.5,
        rng=rng, 
        optimize=True,
        optimize_vars=["alpha"])
echo_layer = keras.layers.RNN(cell, return_sequences=True)
regr = RidgeRegression(rnn_target, num_units, lmb_init)

inp = Input((1, stime))
model = Sequential()
model.add(echo_layer)
model.add(regr)
gd = keras.optimizers.adam(lr)
model.compile(optimizer=gd, loss='mse', metrics=["accuracy"])
cb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
                   write_graph=True, write_images=True)
model.fit(rnn_inputs, rnn_target, epochs=100, batch_size=1, callbacks=[cb])
# print(echo_layer._trainable_weights)

# 
# 
# 
# 
