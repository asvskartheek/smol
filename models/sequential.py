from .model import *

class Sequential(Model):
    def __init__(self,loss):
        super(Sequential,self).__init__(loss)

    def forward(self,X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train_step(self):
        dU = 1 # dL/dL = 1
        dU = self.loss.backward(dU)
        for layer in self.layers[::-1]:
            dU = layer.backward(dU)
            layer.train_step(self.learner)

    def train(self,X,y,lr,learner='sgd',debug=100,epochs=1000):
        self._set_learner(learner)
        for layer in self.layers:
            layer.__set_lr__(lr)

        loss = self.get_loss(X,y)
        for i in range(epochs):
            if (i+1)%debug==0 or i==0:
                print("Loss: {:.4f}, {}/{}".format(loss,i+1,epochs))
            loss = self.get_loss(X,y)
            self.train_step()