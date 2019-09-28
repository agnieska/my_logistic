
# coding: utf-8
#get_ipython().system('pip3 install tqdm')
#get_ipython().system('pip3 install scipy')

import numpy as np
import pandas as pd
from pprint import pprint
import json


######################################################################
# ### Math functions
######################################################################

def sigmoid(z):
    return 1/(1 + np.exp(-z))


# test sigmoid
print("sigmoid de -10 ", sigmoid(-10))

# normalize pour un feature


def centrer_reduire_feature(X):
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


def centrer_reduire_matrix(XXX):
    mean = np.mean(XXX, axis=0)
    stdev = np.std(XXX, axis=0)
    XXX = (XXX - mean)/stdev
    return XXX, mean, stdev

######################################################################
# ### Json functions
######################################################################


def read_json(filename):
    with open(filename, encoding='utf-8') as file:
        data_dict = json.load(file)
    return data_dict


def save_json(data_dict, filename):
    if not filename:
        filename = "myjson.json"
    with open(filename, mode='w', encoding='utf-8') as file:
        json.dump(data_dict, file)


# Loading and cleaning data
###########################################################################################################################
print("\n##################################################################################################################")
print("\n                               LOAD DATA ")
print("\n##################################################################################################################")
print("\n...loading theta training results")
theta_dict = read_json("theta_coeficients.json")
theta_matrix = np.array([theta_dict['Gryffindor'] , theta_dict['Hufflepuff'], theta_dict['Ravenclaw'], theta_dict['Slytherin']])
print("\n\n")
print(theta_matrix)

print("\n...loading test data")
data_test = pd.read_csv("resources/dataset_test.csv")
print("\n\n", data_test.head(10))
data_clean = data_test.fillna(data_test.mean())
column_names_list = list(data_clean.columns)
m = data_clean['Index'].shape[0]
print("Found sample size for test data : ", m)
X2_15 = np.array(data_clean[column_names_list[6:19]])
X2_15_norm, mean, std = centrer_reduire_matrix(X2_15)
X1 = list(data_clean['Best Hand'])
X1 = [0.0 if el == 'Left' else 1.0 for el in X1]
X1 = np.array(X1)
X1_norm, mean, std = centrer_reduire_feature(X1)
X0 = np.ones(m)
X_norm = np.c_[X0, X1_norm, X2_15_norm]
X = X_norm
X_names = ["X_"+str(a) for a in range (0,X.shape[1])]
df = pd.DataFrame(np.around(X_norm, decimals=2))
df.columns = X_names
print("\n\n...Defining X matrix with dimensions :", X.shape, ":\n")
print(df.head(7))


# Prediction
########################################################################################################################

print("\n###########################################################################################################################")
print("\n                              PREDICTION              ")
print("\n############################################################################################################################")
switcher = {0: 'Gryffindor', 2: 'Ravenclaw', 3: 'Slytherin', 1: 'Hufflepuff'}
classProbabilities = []
df = {}
for i in range(0, 4):
    probability = sigmoid(np.dot(X, theta_matrix[i]))
    #probability = np.around(probability, decimals=3)
    #print("\n           RESULT: Probabilities to be classified to",switcher[i],"house are :", np.around(probability, decimals=2)[:10])
    classProbabilities.append(probability)
    #df["Probab_"+switcher[i][:3]] = np.around(probability, decimals=2)
classProbabilities = np.array(classProbabilities)

Classifiers_matrix = []
for i in range(0, 4):
    classifier = [(i+1) if a > 0.4 else 0 for a in classProbabilities[i]]
    Classifiers_matrix.append(classifier)
Classifiers_matrix = np.array(Classifiers_matrix)
Classifiers_list = sum(Classifiers_matrix)
Classifiers_list = np.array(Classifiers_list)
#df["BestClass"] = Classifiers_list

temp = Classifiers_list.copy()
Classifiers_texte = Classifiers_list.copy()
for i in range(0, 4):
    Classifiers_texte = np.where(
        temp == i+1, switcher[i], Classifiers_texte)
df["ClassTexte"] = Classifiers_texte
Classifiers_texte = list(Classifiers_texte)
print("         RESULT: \n", pd.DataFrame(df).head(400))
