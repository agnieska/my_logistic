
system('pip3 install numpy')
system('pip3 install pandas')

import numpy as np
import pandas as pd
from pprint import pprint
import math


def create_describe_dict(data_train, column_names_list) :
    
    print("column names : ", str(column_names_list))
    describe_dict = {}
    for column_name in column_names_list[6:] :
        #values = list(train_dict[column_name].values())
        values = list(data_train[column_name])
        #print(type(values[4]))
        values = [x for x in values if (math.isnan(x) == False)]
        #pprint(values[0:5])
        values.sort()
        values = np.array(values)
        describe_dict[column_name]={}
        count = round(len(values),6)
        somme = round(sum(values),6)
        mean = round(somme/count,6)
        variance = round(sum((values - mean)**2)/count, 6)
        #print(variance)
        std = round(variance **0.5,6)
        #print(std)
        count = int(count)
        q25 = int(count*0.25)
        q50 = int(count*0.5)
        q75 = int(count*0.75)
        describe_dict[column_name]['count'] = count
        #describe_dict[column_name]['sum'] = somme
        describe_dict[column_name]['mean'] = mean
        describe_dict[column_name]['std'] = std
        describe_dict[column_name]['min'] = round(values[0],6)
        describe_dict[column_name]['25%'] = round(values[q25],6)
        describe_dict[column_name]['50%'] = round(values[q50],6)
        describe_dict[column_name]['75%'] = round(values[q75],6)
        describe_dict[column_name]['max'] = round(values[-1],6)
    return describe_dict



def transpose_dict(describe_dict) :
    df = pd.DataFrame(describe_dict)
    print(df.head(10))
    describe_dict = df.transpose().to_dict()
    return describe_dict


def print_columns (column_names_list) :
    first_line = "{:<10s}".format(" ")
    for name in column_names_list[6:11]:
        first_line = first_line +"{:>16s}".format(name[:7])
        #first_line = "{:>10s}".format(name[:10])
    print(first_line)

    first_line = "{:<10s}".format(" ")
    for name in column_names_list[11:16]:
        first_line = first_line +"{:>16s}".format(name[:7])
        #first_line = "{:>10s}".format(name[:10])
    print(first_line)

    first_line = "{:<10s}".format(" ")
    for name in column_names_list[16:]:
        first_line = first_line +"{:>16s}".format(name[:7])
        #first_line = "{:>10s}".format(name[:10])
    print(first_line)


def print_describe_results (column_names_list, measure_names_list) :
    first_line = "{:<10s}".format(" ")
    for name in column_names_list[6:11]:
        first_line = first_line +"{:>16s}".format(name[:7])
        #first_line = "{:>10s}".format(name[:10])
    print(first_line)
    for measure in measure_names_list :
        line = "{:<10s}".format(str(measure))
        #print(s)
        for name in column_names_list[6:11]:
            value = describe_dict[measure][name]
            line = line + "{:16.6f}".format(value)
        print(line)

    first_line = "{:<10s}".format(" ")
    for name in column_names_list[11:16]:
        first_line = first_line +"{:>16s}".format(name[:7])
        #first_line = "{:>10s}".format(name[:10])
    print(first_line)
    for measure in measure_names_list:
        line = "{:<10s}".format(str(measure))
        #print(s)
        for name in column_names_list[11:16]:
            value = describe_dict[measure][name]
            line = line + "{:16.6f}".format(value)
        print(line)

    first_line = "{:<10s}".format(" ")
    for name in column_names_list[16:]:
        first_line = first_line +"{:>16s}".format(name[:7])
        #first_line = "{:>10s}".format(name[:10])
    print(first_line)
    for measure in measure_names_list :
        line = "{:<10s}".format(str(measure))
        #print(s)
        for name in column_names_list[16:]:
            value = describe_dict[measure][name]
            line = line + "{:16.6f}".format(value)
        print(line)


def main () :
    data_train = pd.read_csv("resources/dataset_train.csv")
    data_test = pd.read_csv("resources/dataset_test.csv")

    print(data_train.head(10))
    print(data_test.head(100))
    print("Train sample size : ", str(data_train['Index'].shape))
    print("Test sample size : ", str(data_test['Index'].shape))
    print(data_test.columns)

    column_names_list = list(data_train.columns)
    measure_names_list = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    describe_dict = create_describe_dict(column_names_list)
    print(describe_dict)
    describe_dict = transpose_dict(describe_dict)
    print(describe_dict)
    print_columns (column_names_list)
    print_describe_results (column_names_list, measure_names_list)