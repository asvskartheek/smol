import sys
sys.path.append('/../losses/')
sys.path.append('/../learners/')
from learners.sgd import SGD
from losses.bce_loss import BCELoss
from losses.mse_loss import MSELoss
import numpy as np

class Model:
    def __init__(self,loss):
        self.layers = []
        if loss=='bce':
            self.loss = BCELoss()
        elif loss=='mse':
            self.loss = MSELoss()# TODO: Add here when new Loss is added.
        pass

    def add_layer(self,layer):
        self.layers.append(layer)
        pass

    def get_loss(self,X,y):
        return self.loss.forward(y,self.forward(X)) # TODO: Make sure loss is a scalar

    def _set_learner(self,learner):
        if learner=='sgd':
            self.learner = SGD()# TODO: Add here when new Learner is added.
        pass

    def describe(self):
        self._n_params_ = 0
        description = "-"*35+"\n"
        for layer in self.layers:
            description += layer.describe()+"\n"+"-"*35+"\n"
            self._n_params_ += layer._n_params_
        description += "Total Parameters\t\t"+str(self._n_params_)+"\n"+"-"*35+"\n"
        print(description)
        pass