
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


data_raw = pd.read_csv("resources/dataset_train.csv")
#data_test = pd.read_csv("resources/dataset_test.csv")
data_raw.head(10)
data_raw.mean()

######################################################################
# ### Data cleaining - replace nulls by mean
#####################################################################

data_clean = data_raw.fillna(data_raw.mean())
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
# ### Normalize numerical variables (6-19 column)
######################################################################

# #### Normalize with pandas pour plusieures features a la fois

df = data_clean[column_names_list[6:19]]
data_norm = (df - df.mean()) / (df.max() - df.min())
data_norm



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

# verifier le X
X[:3]

X.shape[0]

# premiere colonne de X (X0)
X[:,0]
# deuxieme colonne de X (hands right=0 left=1)
X[:,1]
# troisieme colonne de X (Arithmacy)
X[:,2]

######################################################################
# ### Definir Y ONE BY ONE
######################################################################

y_raw = list(data_raw['Hogwarts House'])
s = set(y_raw)
print("Houses are: ", s)

y_Gry = np.array([1.0 if el == 'Gryffindor' else 0.0 for el in y_raw ])
y_Huf = np.array([1.0 if el == 'Hufflepuff' else 0.0 for el in y_raw ])
y_Rav = np.array([1.0 if el == 'Ravenclaw' else 0.0 for el in y_raw ])
y_Sly = np.array([1.0 if el == 'Slytherin' else 0.0 for el in y_raw ])
y_Gry.shape, y_Huf.shape, y_Rav.shape, y_Sly.shape

print(y_Gry[0:10], y_Huf[0:10], y_Rav[0:10], y_Sly[0:10])

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
# ### la fonction du cout logistique
######################################################################

# cost 
def cost_log(X, y, theta):
    m = X.shape[0]
    hip = hipothesis_log(X, theta)
    #print("hip=", hip)
    hip[hip == 1] = 0.999 
    #print("hip=", hip)
    loss = y * np.log(hip) + (1-y) * np.log(1-hip)
    #print("loss=", loss)
    cost =  (-1/m) * (np.sum(loss)) 
    return cost


cost_log(X[:,0:15], y_Gry, theta[0:15])

def predict(X, theta):
      return(sigmoid(np.dot(X, theta)))


def cost(X, y, theta):
    return((-1 / X.shape[0]) * np.sum(y * np.log(predict(X, theta)) + (1 - y) * np.log(1 - predict(X, theta))))


cost(X[:,0:15], y_Gry, theta[0:15])

######################################################################
# ### Funtions FIT with cost and visualize Cost
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
        cost = cost_log(X, y, theta)
        J_history.append(cost)
        #J_history.append(cost(X, y, theta))
    return theta, J_history

import matplotlib.pyplot as plt

def visualize_cost(J_history) :
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(J_history)


######################################################################
# ###               ONE BY ONE TRAINING
######################################################################

# learn the Gryffindor House theta parameters  

theta = np.zeros(15)
theta_Gry, J_history_Gry = fit(X, y_Gry, theta, 0.05547, 200000)
print("learned theta for Gryffindor House : \n", theta_Gry)
print("final cost for Gryffindor House = ", J_history_Gry[-1])
visualize_cost(J_history_Gry)
print("... cost evolution for Gryffindor House dispalyed")

# learn the Hufflepuf House theta parameters  
theta = np.zeros(15)
theta_Huf, J_history_Huf = fit(X, y_Huf, theta, 0.05547, 200000)
print("learned theta for Hufflepuf House : \n", theta_Huf)
print("final cost for Hufflepuf House = ", J_history_Huf[-1])
visualize_cost(J_history_Huf)
print("... cost evolution for Hufflepuf House dispalyed")

# learn the Ravenclaw House theta parameters  
theta = np.zeros(15)
theta_Rav, J_history_Rav = fit(X, y_Rav, theta, 0.05547, 200000)
print("theta for Ravenclaw House : \n" , theta_Rav)
print("final cost for Ravenclaw House = ", J_history_Rav[-1])
visualize_cost(J_history_Rav)
print("... cost evolution for Ravenclaw House dispalyed")

# learn the Slytherin House theta parameters  
theta = np.zeros(15)
theta_Sly, J_history_Sly = fit(X, y_Sly, theta, 0.05547, 200000)
print("theta for Slytherin House : \n" , theta_Sly)
print("final cost for Slytherin House = ", J_history_Sly[-1])
visualize_cost(J_history_Sly)
print("... cost evolution for Slytherin House dispalyed")



#####################################################################
# #### Calcul probabilite ONE BY ONE
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
# #### Classification 1 vs 0 ONE BY ONE
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
# ###                    ONE VS ALL TRAINING
#####################################################################

######################################################################
# ### Definir Y one vs all
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