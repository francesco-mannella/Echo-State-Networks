# Copyright 2015 Francesco Mannella (francesco.mannella@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell_impl


class EchoStateRNNCell(rnn_cell_impl.RNNCell):
    """Echo-state RNN cell.
    
    Uses the buildEchoStateWeightsInitializer attribute to build inner weights 
    with echo-state features
    """

    def __init__(self, num_units, decay=0.1, epsilon=1e-10, alpha=0.4, 
                 sparseness=0.0, rng=None, activation=None, reuse=None):
        """
        Args:
            num_units: int, The number of units in the RNN cell.
            decay: float, Decay of the ODE of each unit. Default: 0.1.
            epsilon: float, Discount from spectral radius 1. Default: 1e-10.
            alpha: float [0,1], the proporsion of infinitesimal expansion vs infinitesimal rotation
                of the dynamical system defined by the inner weights
            sparseness: float [0,1], sparseness of the inner weight matrix. Default: 0.
            rng: np.random.RandomState, random number generator. Default: None.
            activation: Nonlinearity to use.  Default: `tanh`.
            reuse: (optional) Python boolean describing whether to reuse 
                variables in an existing scope. If not `True`, and 
                the existing scope already has the given variables, 
                an error is raised.
        """

        # Basic RNNCell initialization
        super(EchoStateRNNCell, self).__init__()
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._reuse = reuse
        self.decay = decay
        self.epsilon = epsilon
        self.alpha = alpha
        self.sparseness = sparseness
        
        # Random number generator initialization
        self.rng = rng
        if rng is None:
            self.rng = np.random.RandomState()
        
        # build initializers for tensorflow variables
        self.W = vs.get_variable('W',initializer = self.buildInputWeigthsInitializer())
        self.U = vs.get_variable('U', initializer = self.buildEchoStateWeightsInitializer())
            
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """ Echo-state RNN: 
            x = x + h*(W*inp + U*f(x) - x). 
        """
        
        new_state = state + self.decay*(
            tf.multiply(inputs, self.W) +
            tf.matmul(self._activation(state), self.U) - state)
        output = self._activation(new_state)
        
        return output, new_state
    
    def buildInputWeigthsInitializer(self):
        """Build a random input weight matrix W 
            
            Returns:
            
            A 1-D tensor representing the 
            input weights to an ESN
            
        """  
        # Input weight tensor initializer
        return self.rng.randn(self._num_units).astype("float32") 
        
    def buildEchoStateWeightsInitializer(self):
        """Build the inner weight matrix initialixer W = u_init so that  
        
            1 - epsilon < rho(W)  < 1,
        
            where 
        
            Wd = decay * W + (1 - decay) * I. 
        
            See Proposition 2 in Jaeger et al. (2007) http://goo.gl/bqGAJu.
            See also https://goo.gl/U6ALDd. 

            Returns:

                A 2-D tensor representing the 
                inner weights of an ESN
             
        """
         
        # Inner weight tensor initializer
        # 1) Build random matrix
        W = self.rng.randn(self._num_units, self._num_units) * \
                (self.rng.rand(self._num_units, self._num_units) < 1 - self.sparseness) 
        # Decompose rotation and translation
        W = self.alpha * (W + W.T) + (1 - self.alpha) * (W - W.T)
        # 2) Normalize to spectral radius 1
        eigvals = np.linalg.eigvals(W)
        W /= np.max(np.abs(eigvals))    
        # 3) Correct spectral radius for leaky units. The iteration 
        #    has to reach this value 
        target = 1.0 - self.epsilon/2.0
        # spectral radius and eigenvalues
        eigvals = np.linalg.eigvals(W)
        x = eigvals.real
        y = eigvals.imag   
        # solve quadratic equations
        a = x**2 * self.decay**2 + y**2 * self.decay**2
        b = 2 * x * self.decay - 2 * x * self.decay**2
        c = 1 + self.decay**2 - 2 * self.decay - target**2
        # just get the positive solutions
        sol = (np.sqrt(b**2 - 4*a*c) - b)/(2*a)
        # and take the minor amongst them
        effective_rho = sol[~np.isnan(sol)].min()
        W *= effective_rho
        # 4 defne the tf variable      

        return W.astype("float32")

