from forword_propagation import *
def predict(parameters,X):
    A2,cache=forword_propagation(X,parameters)
    predictions=np.round(A2)

    return predictions

