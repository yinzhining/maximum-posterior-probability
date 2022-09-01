import numpy as np
def backword_propagation(parameters,cache,X,Y):
    m=X.shape[1]
    w1=parameters["w1"]
    w2=parameters["w2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    Z1=cache["Z1"]
    A1=cache["A1"]
    Z2=cache["Z2"]
    A2=cache["A2"]
    dZ2=A2-Y
    dw2=np.dot(dZ2,A1.T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.multiply(np.dot(w2.T,dZ2),1-np.power(A1,2))
    dw1=np.dot(dZ1,X.T)/m
    db1=np.sum(dZ1,axis=1,keepdims=True)/m

    grads={
        "dw1":dw1,
        "dw2":dw2,
        "db1":db1,
        "db2":db2
    }
    return grads