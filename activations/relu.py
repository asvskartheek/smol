import numpy as np

class Relu:
    def __init__(self):
        pass
    
    def forward(self,X):
        self.out = np.maximum(X,0)
        return self.out

    def backward(self,dU): # Upstream Grad
        self.dU = dU
        mask = (self.out > 0).astype(np.float64)
        self.dD = np.mean(np.multiply(mask,self.dU),axis=0)
        return self.dD