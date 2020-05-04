import numpy as np

class Softmax:
    def __init__(self):
        pass
    
    def forward(self,X):
        exp_inp = np.exp(X)
        self.out = exp_inp/(np.sum(exp_inp,axis=1).reshape(-1,1))
        return self.out

    def backward(self,dU): # Upstream Grad
        self.dU = dU
        mask = np.multiply(self.out,1-self.out)
        self.dD = np.multiply(self.dU,np.mean(mask),axis=0)
        return self.dD