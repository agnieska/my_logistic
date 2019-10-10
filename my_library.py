
import numpy as np
import pandas as pd
import math
import json
import sys
from tqdm import tqdm

######################################################################
# ### Math / Stats functions
######################################################################


def calcul_variance(array):
    count = len(array)
    somme = math.fsum(array)
    mean = np.round(somme/count, 6)
    variance = sum((array - mean)**2)/(count - 1)
    std = np.round(variance ** 0.5, 6)
    return count, mean, std


def calcul_mediane(array):
    l = len(array)
    if l % 2 == 1:
        # print(l, "is impair")
        int_middle = (l + 1) // 2 - 1
        return int_middle, array[int_middle]
    else:
        # print(l, "is pair")
        i = l // 2 - 1
        j = l // 2
        float_middle = (l + 1) / 2 - 1
        middle_value = array[i] + 0.5 * (array[j] - array[i])
        return float_middle, middle_value


def calcul_quantile(array):
    # print("\n\n")
    i50, q50 = calcul_mediane(array)
    if isinstance(i50, int):
        i25, q25 = calcul_mediane(array[:i50+1])
        i75, q75 = calcul_mediane(array[i50:])
    elif isinstance(i50, float):
        ceil = math.ceil(i50)
        i25, q25 = calcul_mediane(np.append(array[:ceil], q50))
        i75, q75 = calcul_mediane(np.append(q50, array[ceil:]))
    else:
        i25 += i75  # c'est rien cette ligne
        print("ERROR boom", "index =", i50, "type=", type(i50))
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


def prepare_dataframe(data, missing, norm):
    column_list = list(data.columns)
    if missing is True:
        data = data.fillna(data.mean())
    if norm is True:
        for name in column_list[6:19]:
            data[name] = (data[name] - data[name].mean()) / (data[name].max() - data[name].min())
    return data, column_list


######################################################################
# ### Linear Regression functions
######################################################################


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def hipothesis(X, theta):
    return (sigmoid(np.dot(X, theta)))


def predict(X, theta):
    return (sigmoid(np.dot(X, theta)))


def calcul_cost(X, y, theta):
    m = X.shape[0]
    hip = hipothesis(X, theta)
    # print("hip=", hip)
    hip[hip == 1] = 0.999
    # print("hip=", hip)
    loss = y * np.log(hip) + (1-y) * np.log(1-hip)
    # print("loss=", loss)
    cost = (-1/m) * (np.sum(loss))
    return cost


def fit(X, y, theta, alpha, num_iters):
    # m : nombre d'enregistrements
    m = X.shape[0]
    J_history = []
    # for _ in range(num_iters):
    for _ in tqdm(range(num_iters)):
        # loss = hipothesis(X, theta) - y
        # gradient = (alpha / m) * np.dot(loss, X))
        # theta = theta - gradient
        theta = theta - (alpha/m) * np.dot((predict(X, theta) - y), X)
        cost = calcul_cost(X, y, theta)
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


def read_csv(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except:
        print("\nERROR: Could not open ", filename)
        sys.exit()

######################################################################
# ### Graphic functions
######################################################################


def print_header(texte):
    l = "###############################################################"
    my_line = "\n"+l+l
    print(my_line)
    print("\n                     "+texte)
    print(my_line)
