import numpy as np
def initialize_parameters(n_x,n_h,n_y):
    """
        w1-(n_h,n_x)
        b1-(n_h,1)
        w2-(n_y,n_h)
        b2-(n_y,1)
    """
    np.random.seed(2)
    w1=np.random.randn(n_h,n_x)
    b1=np.random.randn(n_h,1)
    w2=np.random.randn(n_y,n_h)
    b2=np.random.randn(n_y,1)

    parameters={
        "w1":w1,
        "b1":b1,
        "w2":w2,
        "b2":b2
    }
    return parameters

