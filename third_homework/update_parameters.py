def update_parameters(parameters,grads,learning_rate=1.2):
    w1,w2=parameters["w1"],parameters["w2"]
    b1,b2=parameters["b1"],parameters["b2"]
    dw1,dw2=grads["dw1"],grads["dw2"]
    db1,db2=grads["db1"],grads["db2"]

    w1=w1-learning_rate*dw1
    w2=w2-learning_rate*dw2
    b1=b1-learning_rate*b1
    b2=b2-learning_rate*b2

    parameters={
        "w1":w1,
        "w2":w2,
        "b1":b1,
        "b2":b2
    }
    return parameters