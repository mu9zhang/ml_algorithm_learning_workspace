import numpy as np

def linreg(inputs, targets):
    inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1) # include bias in the input, as do in perceptron
    beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs),inputs)),np.transpose(inputs)),targets)
    outputs = np.dot(inputs,beta)
    #print(shape(beta))
    #print(outputs)
    return beta
