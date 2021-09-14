import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)

def mse(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()

# Weights
w1 = np.random.normal()
w2 = np.random.normal()
w3 = np.random.normal()
w4 = np.random.normal()
w5 = np.random.normal()
w6 = np.random.normal()

# Biases
b1 = np.random.normal()
b2 = np.random.normal()
b3 = np.random.normal()


def n(inputs, bias):
    sum = inputs[0]*weights[0] + inputs[1]*weights[1] + bias
    output = sigmoid(sum)
    return output

def feedforward():
    pass