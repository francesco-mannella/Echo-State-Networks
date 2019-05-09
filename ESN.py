# Copyright 2017 Francesco Mannella (francesco.mannella@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this fileexcept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module implementing the EchoStateRNN Cell.

This module provides the EchoStateRNN Cell, implementing the leaky ESN as 
described in  http://goo.gl/bqGAJu.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
import keras
from keras.constraints import Constraint

class MinMax(Constraint):
    """MinMax weight constraint.

    Constrains the weights incident to each hidden unit
    to be between a lower bound and an upper bound.

    # Arguments
    min_value: the minimum value for the incoming weights.
    max_value: the maximum value for the incoming weights.

    """
    
    def __init__(self, min_value=0.0, max_value=1.0):

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.math.minimum(self.max_value, tf.maximum(self.min_value, w))

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

try:
    py_function = tf.py_function
except AttributeError:
    py_function = tf.contrib.eager.py_func

# to pass to py_func
def np_eigenvals(x):
    return np.linalg.eigvals(x).astype('complex64')

class EchoStateRNNCell(keras.layers.Layer):
    """Echo-state RNN cell.

    """

    def __init__(self, num_units, decay=0.1, epsilon=1e-10, alpha=0.4, 
                 sparseness=0.0, rng=None, activation=None, optimize=False, 
                 optimize_vars=None, reuse=None, **kwargs):
        """
        Args:
            num_units: int, The number of units in the RNN cell.
            input_dim: int, The number of input units to the RNN cell.
            decay: float, Decay of the ODE of each unit. Default: 0.1.
            epsilon: float, Discount from spectral radius 1. Default: 1e-10.
            alpha: float [0,1], the proporsion of infinitesimal expansion vs infinitesimal rotation
                of the dynamical system defined by the inner weights
            sparseness: float [0,1], sparseness of the inner weight matrix. Default: 0.
            rng: np.random.RandomState, random number generator. Default: None.
            activation: Nonlinearity to use.  Default: `tanh`.
            optimize: Python boolean describing whether to optimize rho and alpha. 
            optimize_vars: list containing variables to be optimized
        """

        # Basic RNNCell initialization
        super(EchoStateRNNCell, self).__init__(**kwargs)
        self.num_units = num_units
        self.activation = activation or keras.activations.tanh
        self.decay_init = decay
        self.alpha_init = alpha
        self.epsilon = epsilon
        self.sparseness = sparseness
        self.optimize = optimize
        self.optimize_vars = optimize_vars
        self.rng = rng
        
    def build(self, input_shape):  
        """
            Args:
            input_dim: int, The number of input units to the RNN cell.
        """

        self.input_dim = input_shape[-1]

        # Random number generator initialization
        if self.rng is None:
            self.rng = np.random.RandomState()    
        
        # build initializers for tensorflow variables 
        self.w = self.buildInputWeights()
        self.u = self.buildEchoStateWeights()
        
        self.W = self.add_weight(name='W', shape= self.w.shape, 
                initializer = keras.initializers.Constant(self.w), 
                trainable = False)   
        self.U = self.add_weight(name='U', shape= self.u.shape,  
                initializer =  keras.initializers.Constant(np.zeros(self.u.shape)), 
                trainable = False)

        # alpha and rho default as tf non trainables  
        self.optimize_table = {"alpha": False, 
                               "rho": False, 
                               "decay": False,
                               "sw": False}
        
        if self.optimize == True:
            # Set tf trainables  
            for var in ["alpha", "rho", "decay", "sw" ]:
                if var in self.optimize_vars or self.optimize_vars is None:
                    self.optimize_table[var] = True
        
        # leaky decay
        self.decay = self.add_weight(name='decay', 
                shape=(), initializer = keras.initializers.Constant(self.decay_init), 
                constraint = MinMax(0.1, 0.25), 
                trainable = self.optimize_table["decay"])
        # parameter for dynamic rotation/translation (0.5 means no modifications)
        self.alpha = self.add_weight(name='alpha', 
                shape=(), initializer = keras.initializers.Constant(self.alpha_init), 
                constraint = MinMax(0.0, 1.0), 
                trainable = self.optimize_table["alpha"])
        # the scale factor of the unitary spectral radius (default to no scaling)
        self.rho = self.add_weight(name='Rho', shape=(), 
                initializer = keras.initializers.ones() 
                if self.optimize_table["rho"] else keras.initializers.Constant(1 - self.epsilon), 
                constraint = MinMax(0.8, 5.0), 
                trainable = self.optimize_table["rho"]) 
        # the scale factor of the input weights (default to no scaling) 
        self.sw = self.add_weight(name='sw', 
                shape=(),  initializer = keras.initializers.ones() 
                if self.optimize_table["sw"] else keras.initializers.Constant(1 - self.epsilon),
                constraint = MinMax(0.0001, 10.0), 
                trainable = self.optimize_table["sw"])     
       
        # builds the in
        self.setEchoStateProperty()
        
        self.built = True
        
    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def call(self, inputs, states):
        """ Echo-state RNN: 
            x = x + h*(f(W*inp + U*g(x)) - x). 
        """
        prev_state = states[0]
        state = prev_state + self.decay*(
                self.activation(
                    tf.matmul(inputs, self.W * self.sw) +
                    tf.matmul(self.activation(prev_state), self.U * self.rho_one * self.rho)      
                    )
                - prev_state)

        output = self.activation(state)

        return output, [state]   
         
    def setEchoStateProperty(self):
        """ optimize U to obtain alpha-imporooved echo-state property """

        #self.U = self.set_alpha(self.U, self.u)
        cm = tf.constant(self.u, dtype=tf.float32)
        self.U += 0.5*(self.alpha*(cm + tf.transpose(cm)) + (1 - self.alpha)*(cm - tf.transpose(cm)))
        self.U = self.normalizeEchoStateWeights(self.U)
        self.rho_one = self.buildEchoStateRho(self.U) 
 
    def set_alpha(self, M, m):
        """ Decompose rotation and translation """
       
        cm = tf.constant(m, dtype=tf.float32)
        M += 0.5*(self.alpha*(cm + tf.transpose(cm)) + (1 - self.alpha)*(cm - tf.transpose(cm)))
        return M
        
    def buildInputWeights(self):
        """            
            Returns:
            
            A 1-D tensor representing the 
            input weights to an ESN    
        """  

        # Input weight tensor initializer
        return self.rng.uniform(-1, 1, [self.input_dim, self.num_units]).astype("float32") 
    
    def buildEchoStateWeights(self):
        """            
            Returns:
            
            A 1-D tensor representing the 
            inner weights to an ESN (to be optimized)        
        """    

        # Inner weight tensor initializer
        # 1) Build random matrix
        W = self.rng.randn(self.num_units, self.num_units).astype("float32") * \
                (self.rng.rand(self.num_units, self.num_units) < 1 - self.sparseness) 
        return W
    
    def normalizeEchoStateWeights(self, W):
        # 2) Normalize to spectral radius 1

        eigvals = py_function(np_eigenvals, [W], tf.complex64) 
        W /= tf.reduce_max(tf.abs(eigvals)) 

        return W
              
    def buildEchoStateRho(self, W):
        """Build the inner weight matrix initialixer W  so that  
        
            1 - epsilon < rho(W)  < 1,
        
            where 
        
            Wd = decay * W + (1 - decay) * I. 
        
            See Proposition 2 in Jaeger et al. (2007) http://goo.gl/bqGAJu.
            See also https://goo.gl/U6ALDd. 

            Returns:

                A 2-D tensor representing the 
                inner weights of an ESN      
        """
            
        # Correct spectral radius for leaky units. The iteration 
        #    has to reach this value 
        target = 1.0 - self.epsilon
        # spectral radius and eigenvalues
        eigvals = py_function(np_eigenvals, [W], tf.complex64) 
        x = tf.real(eigvals) 
        y = tf.imag(eigvals)  
        # solve quadratic equations
        a = x**2 * self.decay**2 + y**2 * self.decay**2
        b = 2 * x * self.decay - 2 * x * self.decay**2
        c = 1 + self.decay**2 - 2 * self.decay - target**2
        # just get the positive solutions
        sol = (tf.sqrt(b**2 - 4*a*c) - b)/(2*a)
        # and take the minor amongst them
        effective_rho = tf.reduce_min(sol)
        rho = effective_rho

        return rho


if __name__ == "__main__":

    from keras.layers import RNN, Input, Dense
    from keras.models import Model
    
    input_len = 5
    n_units = 20
    timesteps = 200
    episodes = 1

    inp = Input(shape=(None, input_len))
    cell = EchoStateRNNCell(num_units=n_units, 
                            activation=lambda x:  math_ops.tanh(x), 
                            decay=0.1, 
                            alpha=0.4,
                            epsilon=2.5e-2,
                            rng=np.random.RandomState(), 
                            optimize=True,
                            optimize_vars=["rho", "decay","alpha", "sw"])

    rnn = RNN(cell, return_sequences=True)
    model = Model(inputs = inp, outputs = rnn(inp))

    inp_ = np.zeros((episodes, timesteps, input_len))
    inp_[0,0,:] = np.random.rand(input_len)
    out = model.predict(inp_)
   
    import matplotlib.pyplot as plt
    plt.plot(out[0])
    plt.show()
        
