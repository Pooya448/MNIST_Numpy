# MNIST using pure Numpy 

### Importing Packages

we use MNIST dataset included with Keras, and we also use the one-hot coding method of Keras to transform Y into. a one-hot coded version


```python
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
```

### Loadind the data, Flattening and reshaping the data


```python
# Loading Data

(X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = mnist.load_data()
```


```python
# Reshaping data, Apply one-hot coding

Y_tr_resh = Y_train_orig.reshape(60000, 1)
Y_te_resh = Y_test_orig.reshape(10000, 1)
Y_tr_T = to_categorical(Y_tr_resh, num_classes=10)
Y_te_T = to_categorical(Y_te_resh, num_classes=10)
Y_train = Y_tr_T.T
Y_test = Y_te_T.T
```


```python
# Unrolling data and make it vector shaped, Then flattening the data

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.
```

### Implementing helper functions: relu(z) and softmax(z) and drelu(z)
    - We use softmax activation for output layer and relu for hidden layer, we also use drelu function for calculating thr gradient of the relu function


```python
# Softmax activation function

def softmax(u):
    s = np.exp(u) / np.sum(np.exp(u), axis=0, keepdims=True)
    return s
```


```python
# ReLU activation function

def relu(p):
    r = np.maximum(0, p)
    return r
```


```python
# Derivative of the ReLU function
def drelu(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z
```

### Initializing the Weights and Biases for each layer


```python
# parameters is a dicttionary containing weights and biases and is used throughout the code

parameters = {}

# layer dims in a dictionary containing size of each layer (number of hidden units and input and ouput units)
def initialize_parameters(layer_dims):
    L = len(layer_dims)
    for l in range(1, L):
        
        # initializing W1 and W2 with random numbers and b1 and b2 with zeros.
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * (np.sqrt(2 / layer_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters
```

### Building the deep network Block by Block


```python
# Predict method which runs the final weights and biases, It is used for measuring performance over the test set

def predict(parameters, X_test):
    
    forward_prop(parameters, X_test, activation)
    predictions = np.round(activation["A4"])
    
    return predictions
```


```python
outputs = {}
m = X_train.shape[1]
activation = {}

# The forward propagation method (forward pass)
def forward_prop(parameters, X_train, activation):\
    
    
    
    outputs["Z" + str(1)] = np.dot(parameters["W1"], X_train) + parameters["b1"]
    activation["A" + str(1)] = relu(outputs["Z" + str(1)])
    
    for l in range(2, 4):
        outputs["Z" + str(l)] = np.dot(parameters["W" + str(l)], activation["A" + str(l - 1)]) + parameters["b" + str(l)]
        activation["A" + str(l)] = relu(outputs["Z" + str(l)])
        
    outputs["Z4"] = np.dot(parameters["W4"], activation["A3"]) + parameters["b4"]
    activation["A4"] = softmax(outputs["Z4"])
    
    return outputs, activation
```


```python
# Compute cost method which computes cost using Logistic Loss function (Cross entropy loss)

def compute_cost(activation):
    loss = - np.sum((Y_train * np.log(activation["A4"])), axis=0, keepdims=True)
    cost = np.sum(loss, axis=1) / m
    return cost
```


```python
# Computing all the needed gradient(derivatives), This method is actually the backward pass (Back Propagation)

grad_reg = {}
m = X_train.shape[1]

def back_prop(parameters, outputs, activation):
    
    grad_reg["dZ4"] = (activation["A4"] - Y_train) / m
    for l in range(1, 4):
        grad_reg["dA" + str(4 - l)] = np.dot(parameters["W" + str(4 - l + 1)].T, grad_reg["dZ" + str(4 - l + 1)])
        grad_reg["dZ" + str(4 - l)] = grad_reg["dA" + str(4 - l)] * drelu(outputs["Z" + str(4 - l)])
        
    grad_reg["dW1"] = np.dot(grad_reg["dZ1"], X_train.T)
    grad_reg["db1"] = np.sum(grad_reg["dZ1"], axis=1, keepdims=True)
    
    for l in range(2, 5):
        grad_reg["dW" + str(l)] = np.dot(grad_reg["dZ" + str(l)], activation["A" + str(l - 1)].T)
        grad_reg["db" + str(l)] = np.sum(grad_reg["dZ" + str(l)], axis=1, keepdims=True)
        
    return parameters, outputs, activation, grad_reg
```


```python
# Definning the optimize function which performs gradient descent on the data and fits it to the model

def optimize(grad_reg, learning_rate=0.005):
    
    for i in range(1, 5):
        parameters["W" + str(i)] = parameters["W" + str(i)] - (learning_rate * grad_reg["dW" + str(i)])
        parameters["b" + str(i)] = parameters["b" + str(i)] - (learning_rate * grad_reg["db" + str(i)])
        
    return parameters
```

### Putting all the blocks togather, Forming the neural network model
    - The model it self which uses all the above functions and performs these actions in row:
        1. Calculate output using Forward Propagation
        2. Calculate cost using Compute Cost
        3. Calculate gradients using Back Propagation
        4. Updating Weights and Biases


```python
learning_rate = 0.5
num_iterations = 1000
costs = []

# The whole model

def model(num_iterations, costs, activation):
    
    initialize_parameters([X_train.shape[0], 50, 50, 50, 10])
    
    for l in range(0, num_iterations):
        
        forward_prop(parameters, X_train, activation)
        cost = compute_cost(activation)
        back_prop(parameters, outputs, activation)
        optimize(grad_reg, learning_rate=0.005)
        
        if l % 100 == 0:
            costs.append(cost)
            
        if l % 100 == 0:
            print("Cost after iteration %i: %f" % (l, cost))
            
    return costs, parameters
```

### Running and feeding the data to the model


```python
# Calling the model the feeding it data to train
c, p = model(num_iterations, costs, activation)
```

    Cost after iteration 0: 2.406661
    Cost after iteration 100: 2.121362


### Plotting
    - Now that the model is trained completely, We use the Test set to measure how well the model is trained, and we plot the learning curve


```python
# Calling predict function on the test set using trained parameters of the network

Y_prediction_train = predict(p, X_train)
Y_prediction_test = predict(p, X_test)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
```

    train accuracy: 90.00116666666666 %
    test accuracy: 90.006 %



```python
# Plotting the learning curve
plt.plot(c)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.axis([0, num_iterations/100 + 0.1, 0, 3])
plt.title("Learning rate = " + str(learning_rate))
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-61-7b35fbe5c1a0> in <module>
          1 # Plotting the learning curve
    ----> 2 plt.plot(c)
          3 plt.ylabel('cost')
          4 plt.xlabel('iterations (per hundreds)')
          5 plt.axis([0, num_iterations/100 + 0.1, 0, 3])


    NameError: name 'c' is not defined



```python

```
