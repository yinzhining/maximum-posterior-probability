import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
np.random.seed(1)
import pandas as pd
import numpy as np
data=pd.read_csv("file.csv",usecols=[46,47,50,56,59,60,61,67,68,70,72,73,74,75,78,79],skiprows=[0])
X=np.array(data).T
print(X)
for i in range(0,16):
    if X[i,1]>100:
        X[i,:]=X[i,:]/100
data1=pd.read_csv("file.csv",usecols=[80],skiprows=[0])
Y=np.array(data1).T-1
from layer_sizes import *
from initialize_parameters import *
from forword_propagation import *
from compute_cost import *
from backword_propagation import *
from update_parameters import *
from nn_model import *
from predict import *
parameters=nn_model(X,Y,n_h=4,num_iterations=45000,print_cost=True)

plt.title("Decision Boundary for hidden layer size"+str(4))

predictions=predict(parameters,X)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
