from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle

import numpy as np
import json

clf = pickle.load(open('clf.sav', 'rb'))
X = pickle.load(open('X.sav', 'rb'))
y = pickle.load(open('y.sav', 'rb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)

X = X_test

y_pred_lib = clf.predict(X)

# Constants

intercepts_ = clf.intercepts_ # from clf.intercepts_
coefs_ = clf.coefs_ # from clf.coefs_
hidden_layer_sizes = [100] # From clf.hidden_layer_sizes
n_layers_ = 3 # From clf.n_layers_
n_outputs_ = 1
layer_units = [X.shape[1]] + hidden_layer_sizes + [n_outputs_]

### Save data as JSON

json_X = X.tolist()
json_coefs_ = [x.tolist() for x in coefs_]
json_intercepts_ = [x.tolist() for x in intercepts_]
json_n_layers_ = n_layers_
json_layer_units = layer_units

data = {"coefs_": json_coefs_,
        "intercepts_": json_intercepts_,
        "json_n_layers_": json_n_layers_,
        "json_layer_units": json_layer_units}

json_data = json.dumps(data)

f = open("data.JSON", "a")
f.write(json_data)
f.close()

f = open("xTest.JSON", "a")
f.write(json.dumps(json_X))
f.close()

# PURE PREDICT

import math

def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X

def logistic(X):
    return [1/(1+math.exp(-x)) for x in X]

hidden_activation = relu
output_activation = logistic

# Initialize layers
activations = [X]

for i in range(n_layers_ - 1):
    activations.append(np.empty((X.shape[0],layer_units[i + 1])))

for i in range(n_layers_ - 1):
    activations[i + 1] = np.dot(activations[i], coefs_[i])
    activations[i + 1] += intercepts_[i]
    # For the hidden layers
    if (i + 1) != (n_layers_ - 1):
        activations[i + 1] = hidden_activation(activations[i + 1])
        
activations[i + 1] = output_activation(activations[i + 1])
y_pred = activations[-1]

y_pred = np.array([round(x) for x in y_pred])

print(np.array(y_pred))
print(y_pred_lib)