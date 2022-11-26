import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading dataset
data = load_iris()
# Dividing the dataset into target variable and features
X=data.data
y=data.target

y_new = np.empty(shape = (X.shape[0], 3))
for i in range(0, y.shape[0]):
    if y[i] == [0]:
        y_new[i] = [1,0,0]
    elif y[i] == [1]:
        y_new[i] = [0,1,0]
    elif y[i] == [2]:
        y_new[i] = [0,0,1]

y = y_new

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)

## XOR training data
#X_train = X_test = np.array([
#    [0, 0],
#    [0, 1],
#    [1, 0],
#    [1, 1],
#])

## XOR expected labels
#y_train = y_test = np.atleast_2d([0, 1, 1, 0]).T


learning_rate = 0.1
iterations = 5000
N = y_train.size
 
# Input features
input_size = 4
 
# Hidden layers 
hidden_size = 2
 
# Output layer
output_size = 3
 
results = pd.DataFrame(columns=["mse", "accuracy"])

np.random.seed(10)
 
# Hidden layer
W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))   
 
# Output layer
W2 = np.random.normal(scale=0.5, size=(hidden_size , output_size)) 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()

for itr in range(iterations):    
    
    # Implementing feedforward propagation on hidden layer
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)
    
    # Implementing feed forward propagation on output layer
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    # Calculating the error
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(A2, y_train)
    results=results.append({"mse":mse, "accuracy":acc},ignore_index=True )
    
    # Backpropagation phase
    E1 = A2 - y_train
    dW1 = E1 * A2 * (1 - A2)
    
    E2 = np.dot(dW1, W2.T)
    dW2 = E2 * A1 * (1 - A1)


    # Updating the weights
    W2_update = np.dot(A1.T, dW1) / N
    W1_update = np.dot(X_train.T, dW2) / N
    
    W2 = W2 - learning_rate * W2_update
    W1 = W1 - learning_rate * W1_update

    #print(A2)

results.mse.plot(title="Mean Squared Error")

results.accuracy.plot(title="Accuracy")

Z1 = np.dot(X_test, W1)
A1 = sigmoid(Z1)
 
Z2 = np.dot(A1, W2)
A2 = sigmoid(Z2)
 
acc = accuracy(A2, y_test)
print("Accuracy: {}".format(acc))

plt.show()