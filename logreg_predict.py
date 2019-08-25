
# coding: utf-8
#get_ipython().system('pip3 install tqdm')
#get_ipython().system('pip3 install scipy')

import numpy as np
import pandas as pd
from pprint import pprint
import math
from scipy import log,exp,sqrt,stats
from tqdm import tqdm

#data = pd.read_csv("resources/dataset_train.csv")
data = pd.read_csv("resources/dataset_test.csv")
data.head(10)
data.mean()
data_clean = data.fillna(data.mean())
data_clean

def sigmoid(z):
    # VERSION NUMPY
    #return 1/(1 + np.exp(-z))
    # VERSION SCIPY
    return 1/(1 + exp(-z))

# #####################################################################
# # #### Calcul probabilite
# ######################################################################

# probability_Gry = sigmoid(np.dot(X, theta_Gry))
# print(probability_Gry.shape)
# #print(set(probability_Gry))

# probability_Huf = sigmoid(np.dot(X, theta_Huf))
# print(probability_Huf.shape)
# #print(set(probability_Huf))

# probability_Rav = sigmoid(np.dot(X, theta_Rav))
# print(probability_Rav.shape)
# #print(set(probability_Rav))

# probability_Sly = sigmoid(np.dot(X, theta_Sly))
# print(probability_Sly.shape)
# #print(set(probability_Sly))

# #####################################################################
# # #### Classification 1 vs 0
# #####################################################################


# classifier_Gry = [1 if a > 0.5 else 0 for a in probability_Gry]
# print(sum(classifier_Gry))

# classifier_Huf = [1 if a > 0.5 else 0 for a in probability_Huf]
# print(sum(classifier_Huf))

# classifier_Rav = [1 if a > 0.5 else 0 for a in probability_Rav]
# print(sum(classifier_Rav))


# classifier_Sly = [1 if a > 0.5 else 0 for a in probability_Sly]
# print(sum(classifier_Sly))



# a = 322+536+446+296
# a

# #####################################################################
# # ### One vs all training
# #####################################################################


# coeficients = []

# costs = []
# for c in range(0, 4):
#     theta = np.zeros(15)
#     theta, J_history = fit(X, Y[c], theta, 0.05547, 40000)
#     print(theta[0:20])
#     coeficients.append(theta)
#     costs.append(J_history)
#     #classifiers[c, :] , costs[c, :] = fit(X[:,0:15], y_Gry, theta[0:15], 0.05547, 200000)



# #for J_history in costs :
# visualize_cost(costs[3])
# pprint(coeficients)
# coeficients = np.array(coeficients)
# print(coeficients.shape)


######################################################################
# ###                           PREDICT
#####################################################################


#####################################################################
# ### One vs all Predict
#####################################################################

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
# #### Classifier avec une boucle
#####################################################################

Classifiers = []
for i in range (0,4):
    classifier = [1 if a > 0.5 else 0 for a in classProbabilities[i]]
    pprint(set(classifier))
    pprint(sum(classifier))
    Classifiers.append(classifier)

Classifiers = np.array(Classifiers)
sum(sum(Classifiers))


print(Classifiers.shape)
predictions = []
for raw in Classifiers.T :
    for index, classe in enumerate(raw) :
        if classe == 1 :
            predictions.append(index)
print(np.array(predictions).shape)
print(set(predictions))


predictions = np.array(predictions)
print(predictions.shape)
print(predictions[0:200])

#####################################################################
# #### Classifier avec une sum de matrice
#####################################################################


Classifiers = []
for i in range (0,4):
    classifier = [(i+1) if a > 0.5 else 0 for a in classProbabilities[i]]
    pprint(set(classifier))
    Classifiers.append(classifier)
Classifiers = np.array(Classifiers)
print(sum(Classifiers))
print(sum(sum(Classifiers)))
predictions = sum(Classifiers)-1

predictions = np.array(predictions)
print(predictions.shape)
print(predictions[0:200])

