import numpy as np

def reshape_features(X):
    return X.reshape(X.shape[0],-1).T

def initialize_params(dims):
    return np.zeros((dims,1))

def compute_loss(Yhat, Y):
    return -(Y * np.log(Yhat) +( (1-Y) * np.log((1-Yhat))))

def compute_cost(Yhat, Y, m):
    loss = compute_loss(Yhat,Y)
    return np.sum(loss) / m

# Calculates sigmoid of z
def sigmoid(z):
    return 1 / (1+np.exp(-z))