
#system('pip3 install numpy')
#system('pip3 install pandas')

import numpy as np
import pandas as pd
from pprint import pprint
import math

def calcul_measures_for_one_column(data_train, column_name):
    #values = list(train_dict[column_name].values())
    values = list(data_train[column_name])
    #print(type(values[4]))
    values = [x for x in values if (math.isnan(x) == False)]
    #pprint(values[0:5])
    values.sort()
    values = np.array(values)
    count = round(len(values), 6)
    somme = round(sum(values), 6)
    mean = round(somme/count, 6)
    variance = round(sum((values - mean)**2)/count, 6)
    #print(variance)
    std = round(variance **0.5, 6)
    #print(std)
    count = int(count)
    q25 = int(count*0.25)
    q25 = round(values[q25],6)
    q50 = int(count*0.5)
    q50 = round(values[q50],6)
    q75 = int(count*0.75)
    q75 = round(values[q75],6)
    minimum = round(values[0],6)
    maximum = round(values[-1],6)
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

def create_describe_values(data_train, column_names_list):    
    describe_dict = {}
    for column_name in column_names_list :
        count, mean, std, q25, q50, q75, minimum, maximum = calcul_measures_for_one_column(data_train, column_name)
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
            next_line = next_line + "{:16.6f}".format(value)
        print(next_line)

def print_describe_results (column_names_list, measure_names_list, describe_dict) :
    print_5_columns_describe (column_names_list, measure_names_list, describe_dict, 0, 5)
    print("\n")
    print_5_columns_describe (column_names_list, measure_names_list, describe_dict, 5, 10)
    print("\n")
    print_5_columns_describe (column_names_list, measure_names_list, describe_dict, 10, None)

def main():
    # je recupère les données
    print("\n... loading data\n")
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
    describe_dict = create_describe_values(data_train, column_names_list)
    
    # On fait la transposition du dataset pour pouvoir imprimer plus facilement
    describe_dict = transpose_dict(describe_dict)
    
    # On imprime le describe avec la disposition de 5 colonnes max par page
    print_describe_results (column_names_list, measure_names_list, describe_dict)


main()