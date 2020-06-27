Implementing Echo-State Networks in tensorflow2.

[ESN-definition.ipynb](ESN-definition.ipynb) Describes the mathematics of leaky echo-state 
networks together with an analytical method to correct the spectral radius of the inner weights 
accounting for leakiness and a matricial trick to improve the variance of network states 
during its dynamics.

[ESN.py](ESN.py) Contains the definition of a customized tensorflow RNNCell. 
The inizialization of the weights uses numpy function via [tf.py_function](https://www.tensorflow.org/api_docs/python/tf/py_function) 
because tf.self_adjoint_eigvals only works on self-adjoint matrices.
 
[ESN-usage.ipynb](ESN-usage.ipynb) Contains an example of training on a simple dataset. 

[parametric-sequence-learning.ipynb](parametric-sequence-learning.ipynb) Contains a more complex example. A set of 2Dtrajectories is learned 
and generalization to the whole family of trajectories is tested.


