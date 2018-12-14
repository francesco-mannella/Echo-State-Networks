#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from ESN import EchoStateRNNCell

import multiprocessing as mp


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
    res = np.sin(res) + np.sin(0.51*res)         + np.sin(0.22*res)         + np.sin(0.1002*res)         + np.sin(0.05343*res)
    
    res -= res.min() 
    res /= res.max()
    
    return res

def MSE(P, Y):
    return tf.reduce_mean(tf.squared_difference(P, Y)) 

def NRMSE(P, Y):
    return tf.sqrt( MSE(P, Y)) / (tf.reduce_max(Y) - tf.reduce_min(Y))
   
def ridge_regression(X, Y, num_units, lmb):
    return tf.matmul( tf.matrix_inverse(
        tf.matmul(tf.transpose(X), X) + lmb*tf.eye(num_units)),
              tf.matmul(tf.transpose(X), Y))


# the activation function of the ESN
out_function = lambda x:  math_ops.tanh(x)


# tensorflow graph -------------------------------------------------------------
  
def initialize_ESN(num_units, out_function, rng, decay=0.15, 
                    alpha=0.5, optimize=True):

    cell = EchoStateRNNCell(num_units=num_units, 
                            activation=out_function, 
                            decay=0.15, 
                            alpha=0.5,
                            rng=rng, 
                            optimize=True,
                            optimize_vars=["rho", "decay","alpha", "sw"])
    return cell
    
def ESN_activation(cell, init_state, inputs, stime, num_units):
    states = []
    state = init_state
    for t in range(stime):
        state,_ = cell(inputs=inputs[0,t:(t+1),:], state=state)
        states.append(state)
    outputs = tf.reshape(states, [stime, num_units])   
    return outputs

def do_regression(output_slice, target_slice, num_units, lmb):
    # do the regression on a training subset of the timeseries
    readout_weights = ridge_regression(output_slice, target_slice,
                                       num_units, lmb)
    return readout_weights


def readout_activation(outputs, readout_weights):   
    readouts = tf.matmul(outputs, readout_weights)
    return readouts
    
    
def training(target_slice, readouts_slice, cell, lmb, lr, optimize_alpha=True):
    # calculate the loss over all the timeseries (escluded the beginning
    nrmse = NRMSE(target_slice, readouts_slice) 
    loss = MSE(target_slice, readouts_slice) 

    try: # if optimize == True
        var_list = [cell.rho, cell.decay, cell.sw, lmb]
        if optimize_alpha :
            var_list.append(cell.alpha)
        optimizer = tf.train.AdamOptimizer(lr)
        train = optimizer.minimize(loss, var_list=var_list)
        
    except ValueError: # if optimize == False
        train = tf.get_variable("trial", (), dtype=None)
        
    return nrmse, loss, train
    
def clipping(cell, lmb):
    # clip values
    clip_rho = cell.rho.assign(tf.clip_by_value(cell.rho, 0.8, 5.0))
    clip_alpha = cell.alpha.assign(tf.clip_by_value(cell.alpha, 0.0, 1.0))
    clip_decay = cell.decay.assign(tf.clip_by_value(cell.decay, 0.1, 0.25))
    clip_sw = cell.sw.assign(tf.clip_by_value(cell.sw, 0.0001, 10.0))
    clip_lmb = lmb.assign(tf.clip_by_value(lmb, 0.001, 0.2))
    clip = tf.group(clip_rho, clip_alpha, clip_decay, clip_sw, clip_lmb)   
    return clip

def make_graph(num_units, out_function, stime, lr, optimize_alpha=True):
    
    inputs = tf.placeholder(tf.float32, [batches, stime, num_inputs])
    target = tf.placeholder(tf.float32, [stime, 1])
    init_state = tf.placeholder(tf.float32, [1, num_units])   
    lmb = tf.get_variable("lmb", initializer=lmb_init, 
                        dtype=tf.float32, trainable=True)
    
    cell = initialize_ESN(num_units, out_function, rng)
    outputs = ESN_activation(cell, init_state, inputs, stime, num_units)  
    output_slice = outputs[timewindow_begin:timewindow_end,:]
    target_slice = target[timewindow_begin:timewindow_end,:]
    readout_weights = do_regression(output_slice, target_slice, num_units, lmb)
    readouts = readout_activation(outputs, readout_weights)
    readoutputs_slice = readouts[timewindow_begin:timewindow_end,:]
    nrmse, loss, train = training(target_slice, readoutputs_slice, cell, lmb, lr, 
            optimize_alpha)
    clip = clipping(cell, lmb)
    return (cell, init_state, inputs, outputs, readouts, 
            target, nrmse, lmb, loss, train, clip)


def simulation(args):
        
    rnn_inputs, rnn_target, rnn_init_state, trials, sim, optimize_alpha = args
    
    data = np.zeros([trials, 7])

    tf.reset_default_graph()

    graph = tf.Graph()
    with graph.as_default() as g:
            
        # Build the graph
        
        cell, init_state, inputs, outputs, readouts, target, nrmse, lmb, \
                loss, train, clip = make_graph(num_units, out_function, stime, lr,
                        optimize_alpha)
        
        # Run session 
        with tf.Session(config=config) as session:
            session.run(tf.global_variables_initializer())
            for k in range(trials):
                
                curr_outputs, curr_readouts, curr_loss, curr_nrmse, _ = \
                        session.run([outputs, readouts, loss, nrmse, train], 
                                feed_dict={inputs:rnn_inputs, target: rnn_target,
                                    init_state:rnn_init_state})
                
                session.run(clip)

                rho, alpha, decay, sw, _ =  session.run([cell.rho, cell.alpha, 
                    cell.decay, cell.sw, clip])
                
                data[k, :] = (sim, optimize_alpha, rho, alpha, decay, sw, curr_nrmse)
        print(sim)
    var_names = ('sim', 'optimize', 'rho', 'alpha', 'decay', 'sw', 'nrsme')
    data = {var_name: var_data for var_name, var_data in zip(var_names, data.T) }
    
    return data
                

def simulations(rnn_inputs, rnn_target, rnn_init_state, trials, sims, optimize_alpha):
    
    p = mp.Pool(8)
    
    args = [[rnn_inputs, rnn_target, rnn_init_state, trials, sim, optimize_alpha] 
            for sim in range(sims)]
    data = p.map(simulation, args)
    data = pd.concat([pd.DataFrame(d) for d in data])
    
    return data 


if __name__ == "__main__":

    import pandas as pd

    # Configs ----------------------------------------------------------------------

    # takes only current needed GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # random numbers
    random_seed = np.random.randint(1, 1e5)
    rng = np.random.RandomState(random_seed)


    # Inits ------------------------------------------------------------------------

    batches = 1
    stime = 70
    num_units = 20
    num_inputs = 1
    lr = 0.0001
    lmb_init = 0.0995
    timewindow_begin = 50
    timewindow_end = stime


    # Input -------------------------------------------------------------------------

    rnn_inputs = np.zeros((batches, stime, num_inputs), dtype="float32")
    wave = mult_sines(stime+5).astype("float32")
    rnn_inputs = wave[5:].reshape(1,stime, 1)
    rnn_init_state = np.zeros([batches, num_units], dtype="float32")

    # The output target - mackey-glass ----------------------------------------------

    rnn_target = wave[:-5]
    rnn_target = rnn_target.reshape(stime, 1).astype("float32")
    
    data_alpha = simulations(rnn_inputs, rnn_target, rnn_init_state, trials=20000, sims=15, optimize_alpha=True) 
    data_no_alpha = simulations(rnn_inputs, rnn_target, rnn_init_state, trials=20000, sims=15, optimize_alpha=False) 

    pd.concat([data_alpha, data_no_alpha]).to_csv("data.csv", index=False)


