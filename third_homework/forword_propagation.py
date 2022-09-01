import numpy as np
import sigmoid
def forword_propagation(X,parameters):
    w1=parameters["w1"]
    b1=parameters["b1"]
    w2=parameters["w2"]
    b2=parameters["b2"]

    Z1=np.dot(w1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(w2,A1)+b2
    A2=sigmoid.sigmoid(Z2)
    cache={
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    }
    return A2,cache
