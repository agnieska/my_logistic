
#system('pip3 install numpy')
#system('pip3 install pandas')

import numpy as np
import pandas as pd
from pprint import pprint
import math
from my_library import  calcul_quantile, calcul_variance
'''
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

def calcul_variance (array) :
    count = len(array)
    somme = math.fsum(array)
    mean = np.round(somme/count, 6)
    variance = sum((array - mean)**2)/(count -1)
    std = np.round(variance **0.5, 6)
    return count, mean, std

'''
def calcul_describe_for_one_column(data_train, column_name):
    #values = list(train_dict[column_name].values())
    values = list(data_train[column_name])
    #print(type(values[4]))
    values = [x for x in values if (math.isnan(x) == False)]
    #pprint(values[0:5])
    values.sort()
    values = np.array(values)
    """
    count = len(values)
    somme = sum(values)
    #somme = math.fsum(values)
    mean = np.round(somme/count, 6)
    variance = sum((values - mean)**2)/(count -1)
    #diff = values - mean
    #variance = np.sum(np.power(diff,2))/(count -1)
    #print(variance)
    std = round(variance **0.5, 6)
    #std = np.round(np.sqrt(variance), 6)
    #print(std)
    #print("count =", count)
    q50 = math.floor(count*0.5)
    print("q50 floor =", q50)
    q50 = math.ceil(count*0.5)
    print("q50 ceil =", q50)
    q25 = math.floor(count*0.25)
    q75 = math.floor(count*0.75)
    q25 = np.round(values[q25],6)
    q50 = np.round(values[q50],6)
    q75 = np.round(values[q75],6)
    """
    count, mean, std = calcul_variance (values)
    q25, q50, q75 = calcul_quantile(values)
    mean = np.round(mean, 6)
    std = np.round(std, 6)
    q25 = np.round(q25,6)
    q50 = np.round(q50,6)
    q75 = np.round(q75,6)
    minimum = np.round(values[0],6)
    maximum = np.round(values[-1],6)
    return count, mean, std, q25, q50, q75, minimum, maximum

def append_measures_for_one_column (count, mean, std, q25, q50, q75, minimum, maximum):
    column_describe = {}
    column_describe['count'] = count
    column_describe['mean'] = mean
    column_describe['std'] = std
    column_describe['min'] = minimum
    column_describe['25%'] = q25
    column_describe['50%'] = q50
    column_describe['75%'] = q75
    column_describe['max'] = maximum
    return column_describe

def create_describe_dictionnary(data_train, column_names_list):    
    describe_dict = {}
    for column_name in column_names_list :
        count, mean, std, q25, q50, q75, minimum, maximum = calcul_describe_for_one_column(data_train, column_name)
        describe_dict[column_name] = append_measures_for_one_column(count, mean, std, q25, q50, q75, minimum, maximum)
    return describe_dict

def transpose_dict(describe_dict) :
    df = pd.DataFrame(describe_dict)
    #print(df.head(10))
    describe_dict = df.transpose().to_dict()
    return describe_dict

def print_5_columns_describe (column_names_list, measure_names_list, describe_dict, start, end):
    first_line = "{:<10s}".format(" ")
    for name in column_names_list[start:end]:
        first_line = first_line +"{:>16s}".format(name[:7])
    print(first_line)
    for measure in measure_names_list :
        next_line = "{:<10s}".format(str(measure))
        #print(s)
        for name in column_names_list[start:end]:
            value = describe_dict[measure][name]
            #print("dict value =", value, "type :", type(value))
            next_line = next_line + "{:16.6f}".format(value)
        print(next_line)

def print_describe_results (data, column_names_list, measure_names_list, describe_dict) :
    print_5_columns_describe (column_names_list, measure_names_list, describe_dict, 0, 5)
    print("\n")
    desc = data[column_names_list[:5]].describe()
    print("PANDAS\n", desc)
    print_5_columns_describe (column_names_list, measure_names_list, describe_dict, 5, 10)
    print("\n")
    desc = data[column_names_list[5:10]].describe()
    print("PANDAS\n", desc)
    print_5_columns_describe (column_names_list, measure_names_list, describe_dict, 10, None)
    desc = data[column_names_list[10:]].describe()
    print("PANDAS\n", desc)
    #print_5_columns_describe (column_names_list, measure_names_list, describe_dict, 2, 8)

def main():
    # je recupère les données
    #print("\n... loading data\n")
    data_train = pd.read_csv("resources/dataset_train.csv")
    data_test = pd.read_csv("resources/dataset_test.csv")
    # Je compare le nombre des enregistrements et de colonnes dans train et test
    print("*** TRAIN DATASET *** ")
    print("Train sample size : ", str(data_train['Index'].shape[0]))
    train_column_names_list = list(data_train.columns)
    print("Train columns : ", len(train_column_names_list))
    print("\n*** TEST DATASET *** ")
    print("Test sample size : ", str(data_test['Index'].shape[0]))
    test_column_names_list = list(data_test.columns)
    print("Test columns : ", len(test_column_names_list))
    
    # Je choisis les colonnes pour calculer les statistiques descriptives (seulement les variables avec les donnees numeriques)
    column_names_list = train_column_names_list[6:]
    print("\n*** COLUMNS WITH NUMERICAL VALUES *** ")
    pprint(column_names_list )
    
    # Je crée une liste des statistiques à calculer
    measure_names_list = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    print("\n*** DESCRIPTIVE STATISTICS *** ")
    pprint(measure_names_list)
    print("\n*** DESCRIBE RESULTS *** \n")
    describe_dict = create_describe_dictionnary(data_train, column_names_list)
    
    # On fait la transposition du dataset pour pouvoir imprimer plus facilement
    describe_dict = transpose_dict(describe_dict)
    
    # On imprime le describe avec la disposition de 5 colonnes max par page
    print_describe_results (data_train, column_names_list, measure_names_list, describe_dict)

    
main()