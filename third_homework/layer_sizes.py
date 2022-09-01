def layer_sizes(X,Y):
    n_x=X.shape[0] #输入层特征个数
    n_h=4#隐藏层的节点数
    n_y=Y.shape[0] #输出层特征个数

    return n_x,n_h,n_y
