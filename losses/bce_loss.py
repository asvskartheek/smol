import numpy as np
class BCELoss:
    """
    BCE Loss with predictions as 0 or 1.
    """
    def __init__(self):
        pass

    def forward(self,y_true,y_pred):
        self.y_true = np.array(y_true).astype(np.float)
        self.y_pred = [1e-8 if x==0 else 1-1e-8 if x==1 else x for x in y_pred] # Making sure logs dont blow
        self.y_pred = np.array(y_pred,dtype=np.float)
        loss = np.sum(-np.multiply(self.y_true,np.log(self.y_pred))-np.multiply((1-self.y_true),np.log(1-self.y_pred)))
        loss /= self.y_true.shape[0]
        return loss

    def backward(self,dU): # dU has nothing to do, just a format
        self.dD = (-np.divide(self.y_true,self.y_pred)+np.divide(1-self.y_true,1-self.y_pred))/self.y_true.shape[0]
        self.dD = self.dD.reshape(-1,1)
        self.dD = np.mean(self.dD,axis=0)
        return self.dD