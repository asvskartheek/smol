import sys
sys.path.append('/../activations/')
from activations import *
class Layer:
    def __init__(self,in_dim,out_dim,activation='relu'):
        self.in_dim = in_dim
        self.out_dim = out_dim
        if activation=='relu':
            self.activation = Relu()
        elif activation=='sigmoid':
            self.activation = Sigmoid()
        # TODO: Add here when new Activation is added.
        pass
    
    def __set_lr__(self,lr):
        self.lr = lr

    def describe(self):
        return self.__layer_name__+3*"\t"+str(self._n_params_)