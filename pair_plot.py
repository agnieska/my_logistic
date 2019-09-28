
"""
Faites un script nommé pair_plot.[extension] qui affiche un pair plot ou scatter plot matrix 
(selon la librairie graphique que vous utiliserez). 

À partir de cette visualisation, quelles caractéristiques allez-vous utiliser 
pour entraîner votre prochaine régression logistique ?
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

def prepare_dataframe (filename, missing=True, norm=True):
    data = pd.read_csv(filename)
    column_list = list(data.columns)
    if missing == True :
        data = data.fillna(data.mean())
    # Normaliser ou pas ?
    if norm == True :
        for name in column_list[6:19] :
            data[name] = (data[name] - data[name].mean()) / (data[name].max() - data[name].min())
    return data, column_list


def main ():
    data, column_list = prepare_dataframe ("resources/dataset_train.csv")
    # Select columns with numeric values or not
    df = data[column_list[1:]]
    sns.pairplot(data=df, hue="Hogwarts House")
    plt.show()

main()