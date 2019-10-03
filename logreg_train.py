
# coding: utf-8
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import parser
from pprint import pprint
# from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score

print("\nRESULT: Import completed")

######################################################################
# ### Math functions
######################################################################


def sigmoid(z):
    return 1/(1 + np.exp(-z))

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
    stdev = stdev
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

######################################################################
# ### Model logistic :  Hypothese sigmoidale function
######################################################################


def hipothesis_log(X, theta):
    return (sigmoid(np.dot(X, theta)))


def predict(X, theta):
    return (sigmoid(np.dot(X, theta)))


######################################################################
# ### la fonction du cout logistique
######################################################################

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


######################################################################
# ### Funtions FIT with cost
######################################################################

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


# Loading
###########################################################################################################################
print("\n##################################################################################################################")
print("\n                               LOAD AND CLEAN DATA ")
print("\n##################################################################################################################")
print("\n...loading train data")
try:
    data_raw = pd.read_csv("resources/dataset_train.csv")
except:
    print("         ERROR: File with train data not found")
    sys.exit()


# ### Definir les dimensions du dataset n: nombre de colonnes, m - nombre d'enregistrements, liste des variables
##########################################################################################################################
print("...analysing data ")
try:
    m = data_raw['Index'].shape[0]
    n = data_raw.shape[1]
    # assert(m>100)
    # assert(n>3)
except:
    print("         ERROR: Dataset dimensions to small or impossible to find")
    sys.exit()
print("...renaming columns ")
try:
    column_names_list = list(data_raw.columns)
    column_short_names_list = [ name.replace("Hogwarts", "").replace(" ", "")[:6] for name in column_names_list ]
    #print("                 Column short names are :\n", column_short_names_list)
    data_raw.columns = column_short_names_list
except:
    print("         ERROR: Impossible to find column names in this dataset")
    sys.exit()

# Cleaning
############################################################################################################################
print("...cleaning data : replacing missing values by mean")
try:
    data_clean = data_raw.fillna(data_raw.mean())
    # print(data_clean.head(5))
except:
    print("         ERROR: Replacing missing values by mean failed")
    sys.exit()

print("\n         LOADED: Total sample size : ", m)
print("                 Total Number of columns : ", n)
print("\n\n", data_clean[column_short_names_list[:10]].head(7))
#print(data_raw[column_names_list[5:10]].head(5))
#print(data_raw[column_names_list[10:15]].head(5))
print("\n", data_clean[column_short_names_list[10:]].head(7))
print("\n               Column full names are :\n", column_names_list[:10], "\n", column_names_list[10:])

# Define X
#################################################################################################################################
print("\n\n######################################################################################################################")
print("\n                        DEFINE COLUMNS TO ANALYSE AND LEARN ( X ) ")
print("\n######################################################################################################################")



# Select columns
###############################################################################################################################
print("\n...Taking variables : col_4 Birthday to col_19 Flying")
X3_16 = np.array(data_clean[column_short_names_list[6:19]])

# Date Column "Birthday Day"
###############################################################################################################################
X1 = list(data_clean['Birthd'])
print("...Converting Birthday to age in days ", list(set(X1))[:6])
# convertir X1 en nombre de jours
today = datetime.today()
X1 = [(today - parser.parse(el)).days for el in X1]
X1 = np.array(X1)
#print("         RESULT: Age sample : ", X1)
#X1_norm, mean, std = centrer_reduire_feature(X1)
#print("...Normalizing and adding Age : ",
      #np.around(X1_norm, decimals=2))

# Text Column "Best Hand"
##############################################################################################################################
X2 = list(data_clean['BestHa'])
print("...Converting 'Best Hand' to binary cathegories", set(X2))
# convertir X1 en binaire
X2 = [0.0 if el == 'Left' else 1.0 for el in X2]
X2 = np.array(X2)
#X2_norm, mean, std = centrer_reduire_feature(X2)
#print("...Normalizing and adding Best Hand : ",
      #np.around(X2_norm, decimals=2))

# Concatenate all
##############################################################################################################################
X = np.c_[X1, X2, X3_16]
df = pd.DataFrame(np.around(X, decimals=2))
df.columns = column_short_names_list[4:]
print("         RESULT : Selected and converted X data, dim :", X.shape, ":\n")
print(df.head(7))

# Normalize
##############################################################################################################################
print("\n\n...Normalizing all  variables with center-reduce method")
X_norm, X_means, X_std = centrer_reduire_matrix(X)
print("         RESULT: X Normalized X data, dim",    X_norm.shape, ":\n")
df = pd.DataFrame(np.around(X_norm, decimals=2))
df.columns = column_short_names_list[4:]
print(df.head(7))


# Column for X0
#############################################################################################################################
print("\n\n...Adding a column of ones as X0")
X0 = np.ones(m)
X_norm = np.c_[X0, X_norm]
X_names = ["X_"+str(a) for a in range (0,X_norm.shape[1])]
df = pd.DataFrame(np.around(X_norm, decimals=2))
df.columns = X_names
print("         RESULT : Final X matrix with dimensions :", X_norm.shape, ":\n")
print( df.head(7))

X = X_norm

# Define Y
###############################################################################################################################
print("\n######################################################################################################################")
print("\n                                DEFINE COLUMN TO PREDICT ( Y ) ")
print("\n######################################################################################################################")


# Converting Hogward House to  np.array Y
###############################################################################################################################
print("\n... Converting one Hogwarts House texte column to 4 binary columns")
y_texte = list(data_clean['House'])
print("\n         RESULTS : Number of students classified to Hogward House : ",
      np.array(y_texte).shape[0])
house_names = list(set(y_texte))
house_names.sort()
print("                 Hogward House 4 unique values are: ", house_names)
Y = []
df = {"House": y_texte}
for name in house_names:
    house_binary = [1 if element == name else 0 for element in y_texte]
    Y.append(house_binary)
    df[name[0:3]+"_bin"] = house_binary
Y = np.array(Y)
#print("\n         RESULTS: 4 binary columns for Hougward House are : \n", pd.DataFrame(Y).transpose().head(10))
print("                 4 binary columns for Hougward House are : \n\n",
      pd.DataFrame(df).head(10))


# Trainig
###############################################################################################################################
print("\n#######################################################################################################################")
print("\n                                         ONE VS ALL TRAINING")
print("\n#######################################################################################################################\n")

switcher = {0: 'Gryffindor', 2: 'Ravenclaw', 3: 'Slytherin', 1: 'Hufflepuff'}
theta_matrix = []
costs = []
for c in range(0, 4):
    theta_vector = np.zeros(X.shape[1])
    print("...initialize theta for ", switcher[c])
    theta_vector, J_history = fit(X, Y[c], theta_vector, 0.05547, 4000)
    print("...learned theta for ", switcher[c])
    theta_matrix.append(theta_vector)
    costs.append(J_history)
theta_matrix = np.around(np.array(theta_matrix), decimals=3)

theta_dict = {'Gryffindor': list(theta_matrix[0]),
              'Hufflepuff': list(theta_matrix[1]),
              'Ravenclaw': list(theta_matrix[2]),
              'Slytherin': list(theta_matrix[3]),
              }
df = pd.DataFrame(theta_dict).transpose()
df.columns = X_names
print("\n         THETA RESULTS: \n", df.head(6))
print("\n         SUCCESS: LEARNING COMPLETED !")
theta_dict['means']= list(X_means)
theta_dict['std' ]= list(X_std)
try:
    save_json(theta_dict, "learning_params.json")
    print("                Theta coeficients saved to learning_params.json.")
    print("""         USAGE: Use 'python logreg_precit.py' command  with 'ressources/dataset_test.csv' """)
except:
    print("         ERROR: Problem with saving file")


# Validation
############################################################################################################################################
yes = input("\nDo you want to see the LEARNING VALIDATION ? Y/N\n")
if yes == "Y" or yes == "y":
    print("\n\n\n##########################################################################################################################")
    print("\n                               VALIDATION  and ERROR CALCUL             ")
    print("\n##########################################################################################################################")

    # Calculate probabiities
    ######################################################################################################################################
    print("\n...Calculating probabilities for all 4 classes (Houses) and all students with learned theta coefficients")
    classProbabilities = []
    df = {}
    for i in range(0, 4):
        probability = sigmoid(np.dot(X, theta_matrix[i]))
        #probability = np.around(probability, decimals=3)
        #print("\n           RESULT: Probabilities to be classified to",switcher[i],"house are :", np.around(probability, decimals=2)[:10])
        classProbabilities.append(probability)
        df["Probab_"+switcher[i][:3]] = np.around(probability, decimals=2)
    classProbabilities = np.array(classProbabilities)
    #print("\n           RESULT: Probabilities to be classified to houses are :\n", pd.DataFrame(df).head(10))

    # Classify
    #########################################################################################################################################
    print("...Classifying with numerical values (0,1,2,3) ")
    Classifiers_matrix = []
    for i in range(0, 4):
        classifier = [(i+1) if a > 0.4 else 0 for a in classProbabilities[i]]
        Classifiers_matrix.append(classifier)
    Classifiers_matrix = np.array(Classifiers_matrix)
    Classifiers_list = sum(Classifiers_matrix)
    Classifiers_list = np.array(Classifiers_list)
    df["BestClass"] = Classifiers_list
    #print("         RESULT: ", Classifiers_list[:30])
    #print("         RESULT: \n", pd.DataFrame(df).head(10))

    # Recode
    ########################################################################################################################################
    print("...Recoding learned numerical values (0,1,2,3) into text values ('Gryf','Rav','Sly','Huff')")
    switcher = {0: 'Gryffindor', 2: 'Ravenclaw',
                3: 'Slytherin', 1: 'Hufflepuff'}
    temp = Classifiers_list.copy()
    Classifiers_texte = Classifiers_list.copy()
    for i in range(0, 4):
        Classifiers_texte = np.where(
            temp == i+1, switcher[i], Classifiers_texte)
    df["ClassTexte"] = Classifiers_texte
    Classifiers_texte = list(Classifiers_texte)
    #print("         RESULT: ", Classifiers_texte[:10])
    #print("         RESULT: \n", pd.DataFrame(df).head(10))

    # Calculate error
    ########################################################################################################################################
    print("...Comparing classification results with empirical results ")
    df["Empirical"] = y_texte
    #print("\n      RESULT: Empirical values were: \n", y_texte[:10])
    accuracy = []
    errors = []
    for i in range(0, len(y_texte)):
        if y_texte[i] != Classifiers_texte[i]:
            accuracy.append(0)
            errors.append(i)
        else:
            accuracy.append(1)
    df["Accuracy"] = accuracy
    print("...Calculating accuracy of learning ")
    print("\n", pd.DataFrame(df).head(10))
    print("\n\n     RESULTS:") 
    print("         Good prediction for                ",
          len(y_texte)-len(errors), " students")
    print("         Wrong prediction for                 ", len(errors), " students")
    error_rate = len(errors)/len(y_texte)
    print("         Percentage of wrong predictions:     ", error_rate)
    """ print("         Students witch wrong prediction :", errors)"""
    score = accuracy_score(y_texte, Classifiers_texte)
    print("         Scikitlearn accuracy score:          ", score)
else:
    sys.exit()
