Implementing Echo-State Networks in tensorflow.

[ESN-definition.ipynb](ESN-definition.ipynb) Describes the mathematics of leaky echo-state 
networks together with an analytical method to correct the spectral radius of the inner weights 
accounting for leakiness and a matricial trick to improve the variance of network states 
during its dynamics.

[ESN.py](ESN.py) Contains the definition of a customized tensorflow RNNCell. 
The inizialization of the weights uses numpy and not a tensorflow graph because tf.self_adjoint_eigvals 
only works on self-adjoint matrices.
 
[ESN-usage.ipynb](ESN-usage.ipynb) Contains two examples of tf graphs based on this custom cell. The first is manual graph 
while the second uses tensorflow APIs.
