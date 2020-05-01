import numpy as np

class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self,X):
        self.out = 1./(1+np.exp(-X))
        return self.out

    def backward(self,dU): # Upstream Grad
        self.dU = dU
        mask = np.multiply(self.out,1-self.out)
        self.dD = np.mean(np.multiply(mask,self.dU),axis=0)
        return self.dD