import numpy as np
class SGD:
    def __init__(self):
        pass

    def update(self,parameter,gradient,lr):
        parameter = parameter - lr*gradient
        return parameter