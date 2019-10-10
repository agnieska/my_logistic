
"""
Faites un script nommé pair_plot.[extension] qui affiche un pair plot.

À partir de cette visualisation, quelles features allez-vous
prendre pour entraîner votre régression logistique ?
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from my_library import read_csv, prepare_dataframe


def main(filename):
    warnings.filterwarnings(action='once')
    sns.set()

    if not filename:
        filename = "resources/dataset_train.csv"
    data = read_csv(filename)
    data, column_list = prepare_dataframe(data, missing=False, norm=True)

    # Select columns with numeric values or not
    df = data[column_list[1:]]

    # Answer
    print("\nQUESTION: Quelles caractéristiques allez-vous utiliser")
    print("            pour entraîner la régression logistique?")
    print("\nREPONSE: Toutes sauf celles qui ont une distribution homogene")
    print("           vis a vis la variable à predire (houses).")
    print("           On peut aussi supprimer une des 2 variables correlées.")
    print("\n\n")

    # Plot figures
    sns.pairplot(data=df, hue="Hogwarts House")
    plt.show()


main("")
