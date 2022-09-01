import numpy as np
from layer_sizes import *
from initialize_parameters import *
from forword_propagation import *
from compute_cost import *
from backword_propagation import *
from update_parameters import *
def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    np.random.seed(3)
    n_x=layer_sizes(X,Y)[0]
    n_h=layer_sizes(X,Y)[1]
    n_y=layer_sizes(X,Y)[2]

    parameters=initialize_parameters(n_x,n_h,n_y)
    w1=parameters["w1"]
    w2=parameters["w2"]
    b1=parameters["b1"]
    b2=parameters["b2"]

    for i in range(num_iterations):
        A2,cache=forword_propagation(X,parameters)
        cost=compute_cost(A2,Y,parameters)
        grads=backword_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=0.45)

        if print_cost:
            if i%1000==0:
                print("第",i,"次迭代,成本为"+str(cost))

    return parameters