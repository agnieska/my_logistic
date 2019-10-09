
import numpy as np
import pandas as pd
from pprint import pprint
import math
import json
import sys


######################################################################
# ### Math / Stats functions
######################################################################

def calcul_variance (array) :
    count = len(array)
    somme = math.fsum(array)
    mean = np.round(somme/count, 6)
    variance = sum((array - mean)**2)/(count -1)
    std = np.round(variance **0.5, 6)
    return count, mean, std

def calcul_mediane(array):
    l = len(array)
    if l%2 == 1 :
        print(l, "is impair")
        index = int((l+1)/2)
        print("median impair index", index)
        print("mediane impair value=", array[index])
        print("mediane precedent value=", array[index-1])
        return index, array[index-1]
    else :
        print(l , "is pair")
        i = int(l/2)
        #print ("i = ", i)
        j = i + 1
        #print("j = ", j)
        float_index = (i+j)/2
        print("median pair float index", float_index)
        value = array[i] + (array[j]-array[i])*0.58
        #print("median pair value = ", value)
        return float_index, value

def calcul_quantile (array):
    print("\n\n")
    i50, q50 = calcul_mediane(array)
    print("median index = ", i50, "median value", q50)
    ceil = math.ceil(i50)
    i25, q25 = calcul_mediane(array[:ceil-1])
    print("q25 index = ", i50, "q25 value", q50)
    floor = math.floor(i50)
    i75, q75 = calcul_mediane(array[floor:])
    print("q75 index = ", i50, "q75 value", q50)
    i25 += i75
    return q25, q50, q75


# normalize pour un feature
def center_reduce_feature(X):
    stdev = np.std(X)
    mean = np.mean(X)
    if stdev != 0:
        A = []
        for x in X:
            a = float((x - mean)/stdev)
            A.append(a)
        return np.array(A), stdev, mean
    else:
        return X, stdev, mean

# normalize pour plusieures features a la fois
def center_reduce_matrix_p(XXX, means, stdev):
    XXX = (XXX - means)/stdev
    return np.array(XXX)

def center_reduce_matrix_t(XXX):
    mean = np.mean(XXX, axis=0)
    stdev = np.std(XXX, axis=0)
    XXX = (XXX - mean)/stdev
    return XXX, mean, stdev

######################################################################
# ### Linear Regression functions
######################################################################

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def hipothesis_log(X, theta):
    return (sigmoid(np.dot(X, theta)))

def predict(X, theta):
    return (sigmoid(np.dot(X, theta)))

def cost_log(X, y, theta):
    m = X.shape[0]
    hip = hipothesis_log(X, theta)
    #print("hip=", hip)
    hip[hip == 1] = 0.999
    #print("hip=", hip)
    loss = y * np.log(hip) + (1-y) * np.log(1-hip)
    #print("loss=", loss)
    cost = (-1/m) * (np.sum(loss))
    return cost

def fit(X, y, theta, alpha, num_iters):
    # m : nombre d'enregistrements
    m = X.shape[0]
    J_history = []
    # for _ in tqdm(range(num_iters)):
    for _ in range(num_iters):
        #loss = hipothesis_log(X, theta) - y
        # gradient = (alpha / m) * np.dot(loss, X))
        #theta = theta - gradient
        theta = theta - (alpha/m) * np.dot((predict(X, theta) - y), X)
        cost = cost_log(X, y, theta)
        J_history.append(cost)
    return theta, J_history


######################################################################
# ### Json functions
######################################################################

def read_json(filename):
    try:
        with open(filename, encoding='utf-8') as file:
            data_dict = json.load(file)
        return data_dict
    except:
        print("cant find file ", filename)
        sys.exit

def save_json(data_dict, filename):
    try: 
        if not filename:
            filename = "myjson.json"
        with open(filename, mode='w', encoding='utf-8') as file:
            json.dump(data_dict, file)
    except:
        print("Cant save the file ", filename)
        sys.exit()