
import numpy as np
import pandas as pd
from pprint import pprint
import math
import sys
from my_library import read_csv, calcul_quantile, calcul_variance


def describe_one(data_train, column_name):
    values = list(data_train[column_name])
    values = [x for x in values if (math.isnan(x) is False)]
    values.sort()
    values = np.array(values)
    count, mean, std = calcul_variance(values)
    q25, q50, q75 = calcul_quantile(values)
    mean = np.round(mean, 6)
    std = np.round(std, 6)
    q25 = np.round(q25, 6)
    q50 = np.round(q50, 6)
    q75 = np.round(q75, 6)
    minimum = np.round(values[0], 6)
    maximum = np.round(values[-1], 6)
    return count, mean, std, q25, q50, q75, minimum, maximum


def append_measures(count, mean, std, q25, q50, q75, mi, ma):
    column_describe = {}
    column_describe['count'] = count
    column_describe['mean'] = mean
    column_describe['std'] = std
    column_describe['min'] = mi
    column_describe['25%'] = q25
    column_describe['50%'] = q50
    column_describe['75%'] = q75
    column_describe['max'] = ma
    return column_describe


def create_describe_dictionnary(df_train, column_names_list):
    descr_dict = {}
    for c_name in column_names_list:
        ct, mean, std, q25, q50, q75, mi, ma = describe_one(df_train, c_name)
        descr_dict[c_name] = append_measures(ct, mean, std, q25, q50, q75, mi, ma)
    return descr_dict


def transpose_dict(describe_dict):
    df = pd.DataFrame(describe_dict)
    describe_dict = df.transpose().to_dict()
    return describe_dict


def print_5_columns(column_names, measure_names, describe_dict, start, end):
    first_line = "{:<10s}".format(" ")
    for name in column_names[start:end]:
        first_line = first_line + "{:>16s}".format(name[:7])
    print(first_line)
    for measure in measure_names:
        next_line = "{:<10s}".format(str(measure))
        for name in column_names[start:end]:
            value = describe_dict[measure][name]
            next_line = next_line + "{:16.6f}".format(value)
        print(next_line)


def print_describe_results(data, column_names, measure_names, describe_dict):
    print_5_columns(column_names, measure_names, describe_dict, 0, 5)
    print("\n")
    print_5_columns(column_names, measure_names, describe_dict, 5, 10)
    print("\n")
    print_5_columns(column_names, measure_names, describe_dict, 10, None)
    # print("\n\n PANDAS")
    # print(data[column_names[:5]].describe())
    # print(data[column_names[5:10]].describe())
    # print(data[column_names[10:]].describe())


def main():

    # Ovrir train, voir les dimensions
    if not filename:
        filename = "resources/dataset_train.csv"
    data_train = read_csv(filename)
    print("\n*** TRAIN DATASET *** ")
    print("Train sample size : ", str(data_train['Index'].shape[0]))
    train_column_names = list(data_train.columns)
    print("Train columns : ", len(train_column_names))

    # Choisir les colonnes pour calculer les statistiques descriptives
    column_names = train_column_names[6:]

    # Créer une liste des statistiques à calculer
    measure_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    # Calculer les stats
    describe_dict = create_describe_dictionnary(data_train, column_names)

    # La transposition du dataset pour pouvoir imprimer plus facilement
    descr_dict = transpose_dict(describe_dict)

    # Imprimer le describe avec la disposition de 5 colonnes max par page
    print("\n*** DESCRIBE RESULTS *** \n")
    print_describe_results(data_train, column_names, measure_names, descr_dict)


main()
