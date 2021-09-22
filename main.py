import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)

def mse(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()

# Weights
w = np.random.normal(size=6)

# Biases
b = np.random.normal(size=3)

#Learning rate
learn_rate = .1

def feedforward(x):
    h1 = sigmoid(x[0]*w[0] + x[1]*w[1] + b[0])
    h2 = sigmoid(x[0]*w[2] + x[1]*w[3] + b[1])
    o1 = sigmoid(h1 * w[4] + h2 * w[5] + b[2])
    return o1

# Train

def train(data, all_y_trues):
    for x, y_true in zip(data, all_y_trues):

        # Feed forward
        sum_h1 = x[0]*w[0] + x[1]*w[1] + b[0]
        h1 = sigmoid(sum_h1)

        sum_h2 = x[0]*w[2] + x[1]*w[3] + b[1]
        h2 = sigmoid(sum_h2)

        sum_o1 = h1  *w[4] + h2  *w[5] + b[2]
        o1 = sigmoid(sum_o1)

        y_pred = o1

        # Back propogation derivatives
        ## d_L_d_y_pred
        d_L_d_ypred = -2 * (y_true - y_pred)

        ##Neuron o1
        d_ypred_d_w4 = h1 * d_sigmoid(sum_o1)
        d_ypred_d_w5 = h2 * d_sigmoid(sum_o1)
        d_ypred_d_b3 = d_sigmoid(sum_o1)

        ##Neuron h1
        d_ypred_d_h1 = w[4]*d_sigmoid(sum_o1)
        d_h1_d_w0    = x[0]*d_sigmoid(sum_h1)
        d_h1_d_w1    = x[1]*d_sigmoid(sum_h1)
        d_h1_d_b1    = d_sigmoid(sum_h1)

        ##Neuron h2
        d_ypred_d_h2 = w[5]*d_sigmoid(sum_o1)
        d_h2_d_w2    = x[0]*d_sigmoid(sum_h2)
        d_h2_d_w3    = x[1]*d_sigmoid(sum_h2)
        d_h2_d_b2    = d_sigmoid(sum_h2)

        #Weights updates
        ##Neuron h1
        common_part = learn_rate * d_L_d_ypred * d_ypred_d_h1
        w[0] -= common_part * d_h1_d_w0
        w[1] -= common_part * d_h1_d_w1
        b[0] -= common_part * d_h1_d_b1

        ##Neuron h2
        common_part = learn_rate * d_L_d_ypred * d_ypred_d_h2
        w[2] -= common_part * d_h2_d_w2
        w[3] -= common_part * d_h2_d_w3
        b[1] -= common_part * d_h2_d_b2

        ##Neuron o1
        w[4] -= d_L_d_ypred * d_ypred_d_w4
        w[5] -= d_L_d_ypred * d_ypred_d_w5
        b[0] -= d_L_d_ypred * d_ypred_d_b3


# Weight(-135), Height(-66), Sex
data = np.array([
    [-2, -1],   #Alice   1
    [25, 6],    #Bob     0
    [17, 4],    #Charlie 0
    [-15, -6]   #Diana   1
])

all_y_trues = ([
    1,
    0,
    0,
    1
])

epochs = 1000
for epoch in range(epochs):
    train(data, all_y_trues)
#    if epoch % 10 == 0:
#        y_preds = np.apply_along_axis(feedforward, 1, data)
#        loss = mse(all_y_trues, y_preds)
#        print("Epoch %d loss: %.3f" % (epoch, loss))

emily = np.array([-7, -3]) # 128 фунтов (52.35 кг), 63 дюйма (160 см)
frank = np.array([20, 2])  # 155 pounds (63.4 кг), 68 inches (173 см)
print("Эмили: %.3f" % feedforward(emily)) # 0.951 - Ж
print("Фрэнк: %.3f" % feedforward(frank)) # 0.039 - М