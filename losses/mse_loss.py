import numpy as np
class MSELoss:
    """
    MSE Loss with predictions as 0 or 1.
    """
    def __init__(self):
        pass

    def forward(self,y_true,y_pred):
        self.y_true = np.array(y_true).astype(np.float64)
        self.y_pred = np.array(self.y_pred,dtype=np.float64)
        self.loss = np.sqrt(np.dot(self.y_true-self.y_pred,self.y_true-self.y_pred))
        self.loss /= self.y_true.shape[0]
        return self.loss

    def backward(self,dU): # dU has nothing to do, just a format
        self.dD = (self.y_pred)/(self.y_true.shape[0]*np.sqrt(self.loss))
        self.dD = self.dD.reshape(-1,1)
        self.dD = np.mean(self.dD,axis=0)
        return self.dD