
# coding: utf-8

import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm


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
# ### Load data
######################################################################
print("loading train data")
data_raw = pd.read_csv("resources/dataset_train.csv")
print("raw data loaded : \n", data_raw.head(10))

#####################################################################
# ### Data cleaining - replace nulls by mean
#####################################################################
print("cleaning data : replace missing values by mean")
data_clean = data_raw.fillna(data_raw.mean())
print("data cleanned : \n", data_clean.head(10))

#####################################################################
# ### Liste des variables
#####################################################################

column_names_list = list(data_clean.columns)
print("Found column names :\n", column_names_list)

######################################################################
# ### Definir m
######################################################################

m = data_clean['Index'].shape[0]
print("Found sample size for train data : ", m, type(m))

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
print("Define and Normalize X (columns to analyse in order to find factors")
# Taking numerical variables 
print("Taking numerical variables : col_6 Arithmancy to col_19 Flying")
X2_15 = np.array(data_clean[column_names_list[6:19]])
print("X2-X15 sample:\n")
pd.DataFrame(X2_15).head(6)

print("Normalize numerical variables with center-reduce method")
X2_15_norm, mean, std = centrer_reduire_matrix(X2_15)
print("X2-X15 normalised dimensions :", X2_15_norm.shape)
print("X2-X15 normalized sample:\n")
pd.DataFrame(X2_15_norm).head(6)

#Take text column Best Hand 
print("Take text column Best Hand")
X1 = list(data_clean['Best Hand'])
print("Best Hand text cathegories", set(X1))
# convertir X1 en binaire
X1 = [0.0 if el == 'Left' else 1.0 for el in X1 ]
X1 = np.array(X1)
print("Best Hand converted to binary cathegories", set(X1))
print("Best Hand dimensions 1600 : ", X1.shape)
print( "Best Hand sample : ", X1[:10])
# normalize X1
X1_norm, mean, std = centrer_reduire_feature (X1)
print("Best Hand normalized dimensions 1600 : ", X1_norm.shape)
print("Best Hand normalized sample  : ", X1_norm[:10])

# Adding a column of ones in X
print("Adding a column of ones as X0")
X0 = np.ones(m)
print("X0 shape 1600 : ", X0.shape)
print("X0 sample with ones : ", X0[:10])

print("Concatenate all in one matrix called X_normalized")
X_norm = np.c_[X0, X1_norm, X2_15_norm]
print("X_normalized normalised dimensions :", X_norm.shape)

print("X_normalized normalized sample:\n")
pd.DataFrame(X_norm).head(6)

X = X_norm

######################################################################
# ### Definir Y 
######################################################################
print("\n######################################################################")
print("DEFINING Y  (column to learn and predict) ")
print("######################################################################\n")

print("....Defining Y one by one")
print("Taking texte variable Hogward House")
y_texte = list(data_clean['Hogwarts House'])
print("Dimensions of y_texte : ", np.array(y_texte).shape)
print("y_texte sample : \n", y_texte[0:10])
s = list(set(y_texte))
print("4 Houses are: ", s)
print("Converting one Hogwarts House texte column to 4 binary columns")
y_Gry = np.array([1.0 if el == 'Gryffindor' else 0.0 for el in y_texte ])
y_Huf = np.array([1.0 if el == 'Hufflepuff' else 0.0 for el in y_texte ])
y_Rav = np.array([1.0 if el == 'Ravenclaw' else 0.0 for el in y_texte ])
y_Sly = np.array([1.0 if el == 'Slytherin' else 0.0 for el in y_texte ])
print("y_Gry, Huf, Rav, Sly samples : \n", y_Gry[0:10],"\n", y_Huf[0:10], "\n", y_Rav[0:10], "\n", y_Sly[0:10])
print("y_Gry, Huf, Rav, Sly shapes : \n", y_Gry.shape, y_Huf.shape, y_Rav.shape, y_Sly.shape)
Y = np.array([y_Gry, y_Huf, y_Rav, y_Sly])
print("Dimensions of  Y  : ",  Y.shape)
print("Y sample : \n", Y[0:10])

print("######################################################################\n")
print("Definign Y with a loop")
Y = []
for name in s :  
    Y.append([1.0 if el == name else 0.0 for el in y_texte ])
Y = np.array(Y)
print("2 x dimensions of  Y  : ",  Y.shape)
print("Single column dimension : ", Y[0].shape)
print("Y sample : \n", Y[0:10])


#####################################################################
# ### Verify X, Y
#####################################################################
print("\nVerify X \n")
print("X_norm dimensions :", X.shape)
print("Type of X_norm : ", type(X))
print("X_norm sample :")
#pd.DataFrame(X).head(5)
#print("premiere colonne de X_norm = X0 \n", X[:,0])
#print("deuxieme colonne de X_norm = X1 Hands: left/right\n", X[:,1])
#print("troisieme colonne de X_norm = X2 Arithmacy\n", X[:,2])

print("\nVerify Y \n")
print("Y sample : \n", Y[0:10])
print("Dimensions of y_texte and  Y  : ", np.array(y_texte).shape, Y.shape)
print("Type of y_texte and Y : ", type(y_texte), type(Y))

#####################################################################
# ### Verify l
#####################################################################

l =[]
for i in range (0, Y.shape[0]):
  l.append(sum(Y[0]))
set(l)
print("Verify l liste des cathegories : ", l)
######################################################################
# ### Definir Theta
######################################################################

theta = np.zeros(15)
#theta = np.array([0.1,0.1,0.1,0.1,0.1,0.2,0.3,0.1,0.1,0.2,0.3,0.1,0.1,0.2,0.1])
print("theta zero :", theta)
print("theta type, theta shape", type(theta), theta.shape)
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


#cost_log(X[:,0:15], y_Gry, theta[0:15])

def predict(X, theta):
      return(sigmoid(np.dot(X, theta)))


def cost(X, y, theta):
    return((-1 / X.shape[0]) * np.sum(y * np.log(predict(X, theta)) + (1 - y) * np.log(1 - predict(X, theta))))


#cost(X[:,0:15], y_Gry, theta[0:15])

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
    ax = plt.axes()
    ax.plot(J_history)


######################################################################
# ###               ONE BY ONE TRAINING
######################################################################

# learn the Gryffindor House theta parameters  

theta = np.zeros(15)
theta_Gry, J_history_Gry = fit(X, y_Gry, theta, 0.05547, 4000)
print("learned theta for Gryffindor House : \n", theta_Gry)
print("final cost for Gryffindor House = ", J_history_Gry[-1])
visualize_cost(J_history_Gry)
print("... cost evolution for Gryffindor House dispalyed")

# learn the Hufflepuf House theta parameters  
theta = np.zeros(15)
theta_Huf, J_history_Huf = fit(X, y_Huf, theta, 0.05547, 4000)
print("learned theta for Hufflepuf House : \n", theta_Huf)
print("final cost for Hufflepuf House = ", J_history_Huf[-1])
visualize_cost(J_history_Huf)
print("... cost evolution for Hufflepuf House dispalyed")

# learn the Ravenclaw House theta parameters  
theta = np.zeros(15)
theta_Rav, J_history_Rav = fit(X, y_Rav, theta, 0.05547, 4000)
print("theta for Ravenclaw House : \n" , theta_Rav)
print("final cost for Ravenclaw House = ", J_history_Rav[-1])
visualize_cost(J_history_Rav)
print("... cost evolution for Ravenclaw House dispalyed")

# learn the Slytherin House theta parameters  
theta = np.zeros(15)
theta_Sly, J_history_Sly = fit(X, y_Sly, theta, 0.05547, 4000)
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


#####################################################################
# ### One vs all training
#####################################################################

coeficients = []
costs = []
for c in range(0, 4):
    theta = np.zeros(15)
    theta, J_history = fit(X, Y[c], theta, 0.05547, 4000)
    #print(theta[0:20])
    coeficients.append(theta)
    costs.append(J_history)
    #classifiers[c, :] , costs[c, :] = fit(X[:,0:15], y_Gry, theta[0:15], 0.05547, 2000)



#for J_history in costs :
visualize_cost(costs[3])
#pprint(coeficients)
coeficients = np.array(coeficients)
print(coeficients.shape)



#####################################################################
# #### Calculate probabilities for all Houses
#####################################################################

#probability = sigmoid(np.dot(X, coeficients[0]))
#pprint(probability.shape)
#pprint(probability[0:30])
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

ClassifiersB = []
for i in range (0,4):
    classifier = [1 if a > 0.5 else 0 for a in classProbabilities[i]]
    pprint(set(classifier))
    pprint(sum(classifier))
    ClassifiersB.append(classifier)

ClassifiersB = np.array(ClassifiersB)
print(ClassifiersB.shape)
sum(sum(ClassifiersB))


print(ClassifiersB.shape)
predictionsB = []
for raw in ClassifiersB.T :
    for index, classe in enumerate(raw) :
        if classe == 1 :
            predictionsB.append(index)  
print(np.array(predictionsB).shape)
print(set(predictionsB))


predictionsB = np.array(predictionsB)
print(predictionsB.shape)
print(predictionsB[0:200])

#####################################################################
# #### Classifier 2 : Find  best class with sum of matrix
#####################################################################


ClassifiersM = []
for i in range (0,4):
    classifier = [(i+1) if a > 0.5 else 0 for a in classProbabilities[i]]
    pprint(set(classifier))
    ClassifiersM.append(classifier)
ClassifiersM = np.array(ClassifiersM)
print(sum(ClassifiersM))
print(sum(sum(ClassifiersM)))
predictionsM = sum(ClassifiersM)-1

predictionsM = np.array(predictionsM)
print(predictionsM.shape)
print(predictionsM[0:200])
#print(predictionsM)

#####################################################################
# #### Compare two method of calcul prediction (loop and Matrix)
#####################################################################

check1  = predictionsB - predictionsM[:-1] 
print(predictionsM.shape)
print(predictionsM[0:200])
#print(predictionsM)
check1  = predictionsB - predictionsM[:-1]
#print(check1)
print(sum(check1))
print(set(check1))
errors = []
for index, value in enumerate(check1):
    if value != 0 :
        errors.append(index)
print(len(errors))
#print(errors)
error_rate = len(errors)/len(y_texte)
print("error rate = ", error_rate)

#####################################################################
# #### Compare results with empirical values
#####################################################################

y_recoded = y_texte.copy()
y_recoded = [0 if el == 'Gryffindor' else el for el in y_recoded]
y_recoded = [1 if el == 'Hufflepuff' else el for el in y_recoded]
y_recoded = [2 if el == 'Ravenclaw' else el for el in y_recoded]
y_recoded = [3 if el == 'Slytherin' else el for el in y_recoded]

print(len(y_recoded))
y_recoded = np.array(y_recoded)
print(y_recoded.shape)
#print(y_recoded[0:20])

check2 = predictionsM - y_recoded
print(check2)
print(sum(check2))
print(set(check2))
errors = []
for index, value in enumerate(check2):
    if value != 0 :
        errors.append(index)
print(len(errors))
#print(errors)
error_rate = len(errors)/len(y_recoded)
print("error rate = ", error_rate)

switcher = {0:'Gryffindor', 2:'Ravenclaw', 3:'Slytherin', 1:'Hufflepuff'}
