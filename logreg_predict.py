
# coding: utf-8

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import parser
from pprint import pprint
import sys
from my_library import read_json, save_json, sigmoid, center_reduce_feature,
center_reduce_matrix_p


# Loading and cleaning data
###############################################################################
print("\n####################################################################")
print("\n                        LOAD DATA AND PARAMETERS ")
print("\n####################################################################")
print("\n...loading theta training results")
try:
    theta_dict = read_json("learning_params.json")
except:
    print("         ERROR: File with learning parameters not found")
    sys.exit()
theta_matrix = np.array([
    theta_dict['Gryffindor'],
    theta_dict['Hufflepuff'],
    theta_dict['Ravenclaw'],
    theta_dict['Slytherin']
    ])
means = np.array(theta_dict['means'])
stdev = np.array(theta_dict['std'])
print(theta_matrix)
print("\n...loading test data")
try:
    data_test = pd.read_csv("resources/dataset_test.csv")
except:
    print("         ERROR: File with test data not found")
    sys.exit()

print(data_test.head(10))
data_clean = data_test.fillna(data_test.mean())
column_names_list = list(data_clean.columns)
m = data_clean['Index'].shape[0]
print("Found sample size for test data : ", m)

# prepare data
X1 = list(data_clean['Birthday'])
today = datetime.today()
X1 = [(today - parser.parse(el)).days for el in X1]
X1 = np.array(X1)
X2 = list(data_clean['Best Hand'])
X2 = [0.0 if el == 'Left' else 1.0 for el in X2]
X2 = np.array(X2)
X3_16 = np.array(data_clean[column_names_list[6:19]])
X = np.c_[X1, X2, X3_16]
X_norm = centrer_reduire_matrix_p(X, means, stdev)
X0 = np.ones(m)
X_norm = np.c_[X0, X_norm]
X = X_norm
X_names = ["X_"+str(a) for a in range(0, X.shape[1])]
df = pd.DataFrame(np.around(X_norm, decimals=2))
df.columns = X_names
print("\n\n...Defining X matrix with dimensions :", X.shape, ":\n")
print(df.head(7))


# Prediction
###############################################################################

print("\n####################################################################")
print("\n                          PREDICTION              ")
print("\n####################################################################")
switcher = {0: 'Gryffindor', 2: 'Ravenclaw', 3: 'Slytherin', 1: 'Hufflepuff'}
classProbabilities = []
df = {}
for i in range(0, 4):
    probability = sigmoid(np.dot(X, theta_matrix[i]))
    classProbabilities.append(probability)
classProbabilities = np.array(classProbabilities)

Classifiers_matrix = []
for i in range(0, 4):
    classifier = [(i+1) if a > 0.4 else 0 for a in classProbabilities[i]]
    Classifiers_matrix.append(classifier)
Classifiers_matrix = np.array(Classifiers_matrix)
Classifiers_list = sum(Classifiers_matrix)
Classifiers_list = np.array(Classifiers_list)

temp = Classifiers_list.copy()
Classifiers_texte = Classifiers_list.copy()
for i in range(0, 4):
    Classifiers_texte = np.where(
        temp == i+1, switcher[i], Classifiers_texte)
Classifiers_texte = list(Classifiers_texte)
df["Index"] = data_clean['Index']
df["Hogwarts House"] = Classifiers_texte
pd.DataFrame(df).to_csv("houses.csv", index=False)

print("         RESULT: \n", pd.DataFrame(df).head(20))
print("\n        Predictions saved to houses.csv")
