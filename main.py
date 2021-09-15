import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)

def mse(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()

# Weights
w = np.random.normal(6)

# Biases
b = np.random.normal(3)

def n(w):
    sum = np.dot(x, w) + b
    output = sigmoid(sum)
    return output

# Train

def train():

    # Feed forward
    h1 = sigmiod(x[0]*w[0] + x[1]*w[1] + b[0])
    h2 = sigmiod(x[0]*w[2] + x[1]*w[3] + b[2])
    o1 = sigmiod(h1  *w[4] + h2  *w[5] + b[3])
    y_pred = o1

    # Back propogation
    ## d_L_d_y_pred
    d_L_d_ypred = -2 * (y_true - y_pred)

    ##Neuron o1
    d_ypred_d_w4 = h1 * d_sigmoid(o1)
    d_ypred_d_w5 = h1 * d_sigmoid(o1)

    d_ypred_d_b3