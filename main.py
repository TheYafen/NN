import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(f):
    fx = sigmoid(f)
    return fx*(1-fx)

inputs = np.array([2, 3])
weights = np.array([0, 1])
bias = 0

def n(inputs, bias):
    sum = inputs[0]*weights[0] + inputs[1]*weights[1] + bias
    output = sigmoid(sum)
    return output


def mse(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()

##Neural Network

out_h1 = n(inputs, bias)
out_h2 = n(inputs, bias)

o = n([out_h1, out_h2], bias)

##Training

y_true = [1, 0, 0, 1]
y_pred = [0, 0, 0, 0]
print(o)
