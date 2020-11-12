import numpy as np

def reshape_features(X):
    return X.reshape(X.shape[0],-1).T

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

def compute_loss(Yhat, Y):
    return -(Y * np.log(Yhat) +( (1-Y) * np.log((1-Yhat))))

def compute_cost(Yhat, Y, m):
    loss = compute_loss(Yhat,Y)
    return np.sum(loss) / m

# Calculates sigmoid of z
def sigmoid(z):
    return 1 / (1+np.exp(-z))

def propagate(w, b, X,Y):
    # 1. Forward Propagation
    m = X.shape[1]
    A = sigmoid(w.T @ X)
    cost = compute_cost(A,Y,m)

    # 2. Backward Propagation
    dw = (X @ (A-Y).T)/m
    db = np.sum(A-Y)/m
    cost = np.squeeze(cost)
    grads = {
        'dw' : dw,
        'db' : db
    }

    return grads, cost

def optimize(w, b, X, Y, n_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    n_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []

    for i in range(n_iterations):
        grads, cost = propagate(w,b,X,Y)
        dw = grads['dw']
        db =  grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print (f'Cost after iteration {i}: {cost}')

    params = {
        'w' : w,
        'b': b
    }
    grads = {
        'dw': dw,
        'db': db
    }

    return params, grads, costs

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # initialize parameters with zeros (≈ 1 line of code)
    w,b = initialize_with_zeros(X_train.shape[0])

     # Gradient descent
    parameters, grads, costs = optimize(w,b,X_train, Y_train, num_iterations,learning_rate,print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)           

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d