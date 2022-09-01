import numpy as np


def compute_cost(A2,Y,paramters):
    m=Y.shape[1]
    w1=paramters["w1"]
    w2=paramters["w2"]

    logrpobs=np.multiply(np.log(A2),Y)+np.multiply((1-Y),np.log(1-A2))
    cost=-np.sum(logrpobs)/m
    return cost