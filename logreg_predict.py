
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
def centrer_reduire_feature (X):
    stdev = np.std(X)
    mean = np.mean(X)
    if stdev != 0:
        A = []
        for x in X :
            a = float((x - mean)/stdev)
            A.append(a)
        return np.array(A), stdev, mean
    else : 
        return X, stdev, mean

# normalize pour plusieures features a la fois
def centrer_reduire_matrix(XXX):
    mean = np.mean(XXX,axis=0)
    stdev = np.std(XXX,axis=0)
    XXX = (XXX - mean)/stdev
    return XXX, mean, stdev

######################################################################
# ### Json functions
######################################################################

def read_json (filename) :
    with open(filename, encoding='utf-8') as file:
        data_dict = json.load(file)
    return data_dict  

def save_json (data_dict, filename) :
    if not filename :
        filename = "myjson.json"
    with open(filename, mode='w', encoding='utf-8') as file:
        json.dump(data_dict, file)

######################################################################
# ### Load data
######################################################################

print("loading test data")
data_test = pd.read_csv("resources/dataset_test.csv")
print("data test loaded :\n", data_test.head(10))

######################################################################
# ### Data cleaining - replace nulls by mean
#####################################################################
print("cleaning test data : replace missing values by mean")
data_clean = data_test.fillna(data_test.mean())
print(data_clean.head(20))

######################################################################
# ### Liste des variables
#####################################################################

column_names_list = list(data_clean.columns)
print("Found column names :\n", column_names_list)

######################################################################
# ### Definir m
######################################################################

m = data_clean['Index'].shape[0]
print("Found sample size for test data : ", m, type(m))

######################################################################
# ### Normalize numerical variables (6-19 column)
######################################################################

# #### Normalize with pandas pour plusieures features a la fois

df = data_clean[column_names_list[6:19]]
data_norm = (df - df.mean()) / (df.max() - df.min())
print("Data normalized with pandas :\n", data_norm.head(20))

######################################################################
# ### Definir X 
######################################################################
print("Define X")
# Add a column of ones in X
X0 = np.ones(m)
X0[:10]
X0.shape

X1 = list(data_clean['Best Hand'])
set(X1)

X1 = [0.0 if el == 'Left' else 1.0 for el in X1 ]
X1 = np.array(X1)
X1[:10]
set(X1)

#X2_15 = data_clean[column_names_list[6:19]]
X2_15 = data_norm
X2_15[:10]

#X2_15 = np.array(data_clean[column_names_list[6:19]])
X2_15 = np.array(data_norm)
X2_15[:10]

set(X2_15[:,1])

X = np.c_[X0, X1, X2_15]

######################################################################
# ### Definir Y
######################################################################

y_raw = list(data_raw['Hogwarts House'])
s = set(y_raw)
print("Houses are: ", s)

# recode one vs all with a loop
Y = []
for name in s :  
    Y.append([1.0 if el == name else 0.0 for el in y_raw ])
Y = np.array(Y)
print("Y : ", Y)
print("Y shape : ", Y.shape)
print("Y de zero : ", Y[0])

######################################################################
# ###   Load theta training results
#####################################################################
theta_Gry = [-3.9722245,  0.6466751, -4.38681363, 1.08499541, -6.33317829, -0.94957602,
  4.8350571 , 0.09037308, 4.57397408, -2.65027486, -3.73571991, -0.94770564,
  0.55564161, -0.72931256,  4.76197122]

theta_Huf = [-2.43380345,  0.12385705,  2.00508575, 5.12481853,  7.13518874, -5.86562983,
  1.99679449, -4.89929829, -6.45421237,  3.48320727,  1.99878244, -1.27843303,
 -0.23649972,  0.06063479, -2.7376722, ]

theta_Rav = [-2.52263809, -0.20594758,  1.75358726, -3.42685654,  3.43223144,  3.3144334,
  1.19111064,  4.238581,    4.62289409,  0.28998038, -0.77526696, -0.1024706,
  0.4070898,   4.83535708,  0.74307937]

theta_Sly = [-3.66138804, -0.35445388,  0.15595263, -3.46998332, -2.94613117,  3.87244453,
 -6.90906571, -1.59053254, -3.17071882, -0.23835901,  3.85240142,  2.99487082,
 -0.94066588, -3.66170159, -2.50883135]

coeficients = np.array(
[-2.26799935,  0.0992041 ,  1.05650397,  5.44707833,  5.8868473 ,
       -5.6359005 ,  2.2548923 , -3.78394826, -6.03727584,  3.30425744,
        3.20755539, -1.39838838, -0.08227454, -0.19168908, -2.5643104 ],
 [-3.6008707 ,  0.52086248, -1.53524942,  1.08702249, -3.95523762,
       -1.0612541 ,  2.29110342,  0.21486748,  4.09650304, -3.55194895,
       -4.16178004, -0.84902982,  0.14478133, -1.2653804 ,  4.23126941],
[-3.4982832 , -0.3027943 , -0.05872397, -3.5238589 , -3.59539725,
        3.60056079, -6.22275092, -1.92559896, -3.02418201,  0.69593325,
        1.90624177,  2.5864709 , -0.39734004, -3.07443707, -2.50380591],
 [-2.50639018, -0.19583837,  0.86200821, -3.19342232,  2.60047424,
        3.19458757,  1.84019135,  4.57164359,  4.32685853,  0.17382821,
       -0.0527677 , -0.27677494,  0.2625099 ,  4.73208425,  0.39829606])

pprint(coeficients)
coeficients = np.array(coeficients)
print(coeficients.shape)

#####################################################################
# ### Define logarithmic function
#####################################################################

def sigmoid(z):
    # VERSION NUMPY
    #return 1/(1 + np.exp(-z))
    # VERSION SCIPY
    return 1/(1 + exp(-z))


#####################################################################

####                    ONE VS ALL PREDICT                       ####

#####################################################################


#####################################################################
# #### Calculate probabilities for all Houses
#####################################################################

probability = sigmoid(np.dot(X, coeficients[0]))
pprint(probability.shape)
pprint(probability[0:30])
pprint(set(probability))

#classProbabilities = sigmoid(X * classifiers.T)
classProbabilities = []
for i in range(0,4):
    probability = sigmoid(np.dot(X, coeficients[i]))
    print(probability.shape)
    print(probability[0:30])
    classProbabilities.append(probability)

classProbabilities = np.array(classProbabilities)
classProbabilities.shape
print(classProbabilities[1].shape)

#####################################################################
# #### Classifier 1 : Find  best class with a loop
#####################################################################

Classifiers1 = []
for i in range (0,4):
    classifier = [1 if a > 0.5 else 0 for a in classProbabilities[i]]
    pprint(set(classifier))
    pprint(sum(classifier))
    Classifiers1.append(classifier)

Classifiers1 = np.array(Classifiers1)
sum(sum(Classifiers1))


print(Classifiers1.shape)
predictions1 = []
for raw in Classifiers1.T :
    for index, classe in enumerate(raw) :
        if classe == 1 :
            predictions1.append(index)
print(np.array(predictions1).shape)
print(set(predictions1))


predictions1 = np.array(predictions1)
print(predictions1.shape)
print(predictions1[0:200])

#####################################################################
# #### Classifier 2 : Find  best class with sum of matrix
#####################################################################


Classifiers2 = []
for i in range (0,4):
    classifier = [(i+1) if a > 0.5 else 0 for a in classProbabilities[i]]
    pprint(set(classifier))
    Classifiers2.append(classifier)
Classifiers2 = np.array(Classifiers2)
print(sum(Classifiers2))
print(sum(sum(Classifiers2)))
predictions2 = sum(Classifiers2)-1

predictions2 = np.array(predictions2)
print(predictions2.shape)
print(predictions2[0:200])

