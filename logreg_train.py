
# coding: utf-8
#get_ipython().system('pip3 install tqdm')
#get_ipython().system('pip3 install scipy')

import numpy as np
import pandas as pd
from pprint import pprint
import math
from scipy import log,exp,sqrt,stats
from tqdm import tqdm
import statistics


data = pd.read_csv("resources/dataset_train.csv")
#data_test = pd.read_csv("resources/dataset_test.csv")
data.head(10)
data.mean()
data_clean = data.fillna(data.mean())
data_clean


######################################################################
# ### Liste des variables
#####################################################################

column_names_list = list(data_clean.columns)
column_names_list

######################################################################
# ### Definir m
######################################################################

m = data_clean['Index'].shape[0]
m
type(m)

######################################################################
# ### Normalize
######################################################################

# #### Normalize with pandas pour plusieures features a la fois

df = data_clean[column_names_list[6:19]]
data_norm = (df - df.mean()) / (df.max() - df.min())
data_norm

###   Normalize with python pour un feature
def centrer_reduire_feature (X):
    stdev = statistics.stdev(X)
    mean = statistics.mean(X)
    A = []
    for x in X :
        a = float((x - mean)/stdev)
        A.append(a)
    return np.array(A), stdev, mean

#####   Normalize with python pour plusieures features a la fois
def centrer_reduire_matrix(XXX):
    mean = XXX.mean(axis=0)
    stdev = XXX.std(axis=0)
    XXX = (XXX - mean)/stdev
    return XXX, mean, stdev



######################################################################
# ### Definir X 
######################################################################

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
X[:3]

X.shape[0]

# premiere colonne de X (X0)
X[:,0]
# deuxieme colonne de X (hands right=0 left=1)
X[:,1]
# troisieme colonne de X (Arithmacy)
X[:,2]

######################################################################
# ### Definir Y
######################################################################

y_raw = list(data['Hogwarts House'])
s = set(y_raw)
s

y = np.array([ord(a[0]) for a in y_raw])
y[:10]


y_Gry = np.array([1.0 if el == 'Gryffindor' else 0.0 for el in y_raw ])
y_Huf = np.array([1.0 if el == 'Hufflepuff' else 0.0 for el in y_raw ])
y_Rav = np.array([1.0 if el == 'Ravenclaw' else 0.0 for el in y_raw ])
y_Sly = np.array([1.0 if el == 'Slytherin' else 0.0 for el in y_raw ])
y_Gry.shape, y_Huf.shape, y_Rav.shape, y_Sly.shape

y_Gry[0:10], y_Huf[0:10], y_Rav[0:10], y_Sly[0:10]

Y = []
for name in s :  
    Y.append([1.0 if el == name else 0.0 for el in y_raw ])
Y = np.array(Y)
Y
# Y = Y.T
# Y

Y.shape
Y[0]

######################################################################
# ### Definir Theta
######################################################################

theta = np.zeros(15)
#theta = np.array([0.1,0.1,0.1,0.1,0.1,0.2,0.3,0.1,0.1,0.2,0.3,0.1,0.1,0.2,0.1])

X.shape, Y.shape, theta.shape
X.shape[0]

l =[]
for i in range (0, Y.shape[0]):
  l.append(sum(Y[0]))
set(l)

type(X), type(y), type(theta)

######################################################################
# ### Model lineaire :  Hypothese y=ax+b
######################################################################


def hipothesis_linear(X, theta):
    #return theta[0] + theta[1] * X
    return np.dot(X, theta)

# test hipothesis
h = hipothesis_linear(X[:,0:2], theta[0:2])
h.shape
set(h)
h[:10]

######################################################################
# ### Model logistic :  Hypothese sigmoidale
######################################################################

def sigmoid(z):
    # VERSION NUMPY
    #return 1/(1 + np.exp(-z))
    # VERSION SCIPY
    return 1/(1 + exp(-z))

# test sigmoid
sigmoid(-10)


def hipothesis_log(X, theta):  
    return (sigmoid(np.dot(X, theta)))
    # returns a 100 x 1 matrix


# test hipothesis
h = hipothesis_log(X[:,0:15], theta[0:15])
h.shape

######################################################################
# ### Tester la fonction du cout logistique
######################################################################

def cost_linear(X, y, theta):
    m = X.shape[0]
    print("m=", m)
    loss = hipothesis_linear(X, theta) - y
    print("loss=", loss)
    print('loss shape=', loss.shape)
    print("loss values=", set(loss))
    #mean_loss = np.sum((loss ** 2) ** 0.5)/ m
    #print("mean_loss=", mean_loss)
    loss2 = loss ** 2
    print("loss2=", loss2)
    print('loss shape=', loss2.shape)
    c = np.sum(loss2)
    print("c=" , c)
    cost = (np.sum(loss ** 2)) / (2 * m) 
    print("cost", cost)
    return cost


cost_linear(X[:,0:15], y_Gry, theta[0:15])



# cost 2018
def cost_log_2018(X, y, theta):
    m = X.shape[0]
    hip = hipothesis_log(X, theta)
    hip[hip == 1] = 0.999 
    #print("hip=", hip)
    c1 = np.dot(y.T, np.log(hip))
    print("cost Y=", c1)
    c2 = np.dot((1-y).T, np.log(1-hip))
    print("cost 1-Y=", c2)
    cost = (-1/m) *  (c1 + c2)
    return cost



cost_log_2018(X[:,0:15], y_Gry, theta[0:15])



# cost 2019
def cost_log_2019(X, y, theta):
    m = X.shape[0]
    hip = hipothesis_log(X, theta)
    #print("hip=", hip)
    hip[hip == 1] = 0.999 
    #print("hip=", hip)
    loss = y * np.log(hip) + (1-y) * np.log(1-hip)
    #print("loss=", loss)
    cost =  (-1/m) * (np.sum(loss)) 
    return cost


cost_log_2019(X[:,0:15], y_Gry, theta[0:15])

def predict(X, theta):
      return(sigmoid(np.dot(X, theta)))


def cost(X, y, theta):
    return((-1 / X.shape[0]) * np.sum(y * np.log(predict(X, theta)) + (1 - y) * np.log(1 - predict(X, theta))))


cost(X[:,0:15], y_Gry, theta[0:15])

######################################################################
# ### Training
######################################################################


def fit(X, y, theta, alpha, num_iters):
    # Initialiser certaines variables utiles
    m = X.shape[0]
    J_history = []
    for _ in tqdm(range(num_iters)):
    #for _ in range(num_iters):
        #loss = hipothesis_log(X, theta) - y
        #gradient = (alpha / m) * (np.dot(loss, X))
        #theta = theta - gradient
        theta = theta - (alpha/m) * np.dot((predict(X, theta) - y), X)
        cost = cost_log_2019(X, y, theta)
        J_history.append(cost)
        #J_history.append(cost(X, y, theta))
    return theta, J_history



######################################################################
###          visualize cost
######################################################################
import matplotlib.pyplot as plt

def visualize_cost(J_history) :
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(J_history)

theta = np.zeros(15)
theta_Gry, J_history = fit(X, y_Gry, theta, 0.05547, 200000)

theta_Gry

J_history[-1]

visualize_cost(J_history)

theta = np.zeros(15)
theta_Huf, J_history = fit(X, y_Huf, theta, 0.05547, 200000)
print(theta_Huf)
print(J_history[-1])
visualize_cost(J_history)



theta = np.zeros(15)
theta_Rav, J_history = fit(X, y_Rav, theta, 0.05547, 200000)
print(theta_Rav)
print(J_history[-1])
visualize_cost(J_history)



theta = np.zeros(15)
theta_Sly, J_history = fit(X, y_Sly, theta, 0.05547, 200000)
print(theta_Sly)
print(J_history[-1])
visualize_cost(J_history)




#####################################################################
# #### Calcul probabilite
######################################################################

probability_Gry = sigmoid(np.dot(X, theta_Gry))
print(probability_Gry.shape)
#print(set(probability_Gry))

probability_Huf = sigmoid(np.dot(X, theta_Huf))
print(probability_Huf.shape)
#print(set(probability_Huf))

probability_Rav = sigmoid(np.dot(X, theta_Rav))
print(probability_Rav.shape)
#print(set(probability_Rav))

probability_Sly = sigmoid(np.dot(X, theta_Sly))
print(probability_Sly.shape)
#print(set(probability_Sly))

#####################################################################
# #### Classification 1 vs 0
#####################################################################


classifier_Gry = [1 if a > 0.5 else 0 for a in probability_Gry]
print(sum(classifier_Gry))

classifier_Huf = [1 if a > 0.5 else 0 for a in probability_Huf]
print(sum(classifier_Huf))

classifier_Rav = [1 if a > 0.5 else 0 for a in probability_Rav]
print(sum(classifier_Rav))


classifier_Sly = [1 if a > 0.5 else 0 for a in probability_Sly]
print(sum(classifier_Sly))



a = 322+536+446+296
a

#####################################################################
# ### One vs all training
#####################################################################


coeficients = []
costs = []
for c in range(0, 4):
    theta = np.zeros(15)
    theta, J_history = fit(X, Y[c], theta, 0.05547, 40000)
    print(theta[0:20])
    coeficients.append(theta)
    costs.append(J_history)
    #classifiers[c, :] , costs[c, :] = fit(X[:,0:15], y_Gry, theta[0:15], 0.05547, 200000)



#for J_history in costs :
visualize_cost(costs[3])
pprint(coeficients)
coeficients = np.array(coeficients)
print(coeficients.shape)



# ######################################################################
# # ###                           PREDICT
# #####################################################################



# #####################################################################
# # ### One vs all Predict
# #####################################################################


# probability = sigmoid(np.dot(X, coeficients[0]))
# pprint(probability.shape)
# pprint(probability[0:30])
# pprint(set(probability))

# #classProbabilities = sigmoid(X * classifiers.T)
# classProbabilities = []
# for i in range(0,4):
#     probability = sigmoid(np.dot(X, coeficients[i]))
#     print(probability.shape)
#     print(probability[0:30])
#     classProbabilities.append(probability)

# classProbabilities = np.array(classProbabilities)
# classProbabilities.shape
# print(classProbabilities[1].shape)

# #####################################################################
# # #### Classifier avec une boucle
# #####################################################################

# Classifiers = []
# for i in range (0,4):
#     classifier = [1 if a > 0.5 else 0 for a in classProbabilities[i]]
#     pprint(set(classifier))
#     pprint(sum(classifier))
#     Classifiers.append(classifier)

# Classifiers = np.array(Classifiers)
# sum(sum(Classifiers))


# print(Classifiers.shape)
# predictions = []
# for raw in Classifiers.T :
#     for index, classe in enumerate(raw) :
#         if classe == 1 :
#             predictions.append(index)
# print(np.array(predictions).shape)
# print(set(predictions))


# predictions = np.array(predictions)
# print(predictions.shape)
# print(predictions[0:200])

# #####################################################################
# # #### Classifier avec une sum de matrice
# #####################################################################


# Classifiers = []
# for i in range (0,4):
#     classifier = [(i+1) if a > 0.5 else 0 for a in classProbabilities[i]]
#     pprint(set(classifier))
#     Classifiers.append(classifier)
# Classifiers = np.array(Classifiers)
# print(sum(Classifiers))
# print(sum(sum(Classifiers)))
# predictions = sum(Classifiers)-1

# predictions = np.array(predictions)
# print(predictions.shape)
# print(predictions[0:200])

