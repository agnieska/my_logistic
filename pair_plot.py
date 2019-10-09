
"""
Faites un script nommé pair_plot.[extension] qui affiche un pair plot.

À partir de cette visualisation, quelles features allez-vous
prendre pour entraîner votre régression logistique ?
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def prepare_dataframe(filename, missing=True, norm=True):
    data = pd.read_csv(filename)
    column_list = list(data.columns)
    if missing is True:
        data = data.fillna(data.mean())
    # Normaliser ou pas ?
    if norm is True:
        for name in column_list[6:19]:
            data[name] = (data[name] - data[name].mean()) /
            (data[name].max() - data[name].min())
    return data, column_list


def main():
    warnings.filterwarnings(action='once')
    data, column_list = prepare_dataframe(
        "resources/dataset_train.csv",
        missing=False,
        norm=True
        )
    # Select columns with numeric values or not
    df = data[column_list[1:]]
    print("\nQUESTION: Quelles caractéristiques allez-vous utiliser")
    print("            pour entraîner la régression logistique?")
    print("\nREPONSE: Toutes sauf celles qui ont une distribution homogene")
    print("           vis a vis la variable à predire (houses).")
    print("           On peut aussi supprimer une des 2 variables correlées.")
    print("\n\n")
    sns.pairplot(data=df, hue="Hogwarts House")
    plt.show()
    
main()
