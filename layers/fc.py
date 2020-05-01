import numpy as np
from .layer import *

class FC(Layer):
    """
    Fully Connected Layer, initialized with Xavier Initialisation.
    """
    def __init__(self,in_dim,out_dim,activation='relu'):
        super().__init__(in_dim,out_dim,activation='relu')

        self.__layer_name__ = "FC Layer"
        # Weird Wt Init.
        self.W = np.random.randn(self.in_dim,self.out_dim)/self.in_dim
        if activation=='relu':
            self.W /= 2
        self.b = np.random.randn(self.out_dim,)
        self._n_params_ = self.in_dim*self.out_dim + self.out_dim # W,b

    def forward(self,X): # X is input
        self.input = X
        self.out = self.activation.forward(np.dot(X,self.W)+self.b)
        return self.out

    def backward(self,dU): # dU is upstream gradient
        self.dU = self.activation.backward(dU)
        self.dB = self.dU
        self.dD = np.mean(self.W.dot(self.dU),axis=0)
        self.dW = np.empty_like(self.W)
        for row in self.input:
            self.dU = self.dU.reshape(-1,1)
            row = row.reshape(-1,1)
            self.dW += np.dot(row,self.dU.T)/ self.input.shape[0]
        return self.dD

    def train_step(self,learner):
        self.learner = learner
        self.W = self.learner.update(self.W,self.dW,self.lr)
        self.b = self.learner.update(self.b,self.dB,self.lr)