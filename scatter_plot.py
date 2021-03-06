
"""
Faites un script nommé scatter_plot.[extension] qui affiche un scatter plot
répondant à la question suivante:
Quelles sont les deux features qui sont semblables ?
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from my_library import read_csv, prepare_dataframe


def sns_scatter(df, col_name1, col_name2, col_name3, fig_nb):
    plt.rc('font', size=12)
    plt.rc('axes', labelsize=12)
    plt.figure(fig_nb)
    ax = sns.scatterplot(data=df, x=col_name1, y=col_name2, hue=col_name3)
    ax.set_title(col_name1+" by "+col_name2)
    # plt.show()


def plt_scatters(df, variables, fig_nb):
    n_rows = len(variables)
    n_cols = len(variables)
    plt.rc('font', size=2)
    plt.rc('axes', labelsize=10)
    fig = plt.figure(fig_nb, figsize=(10, 10), dpi=100)
    for i, var_name1 in enumerate(variables):
        for j, var_name2 in enumerate(variables):
            number = i*len(variables)+j+1
            ax = fig.add_subplot(n_rows, n_cols, number)
            ax.scatter(
                x=df[var_name1],
                y=df[var_name2],
                color=np.random.rand(3),
                s=0.5
                )
            if (number-1) % n_cols == 0:
                plt.ylabel(var_name1[:6])
            if number > (n_rows-1) * n_cols:
                plt.xlabel(var_name2[:6])
    fig.tight_layout()  # Improves appearance a bit.
    # plt.show()


def main(filename):
    warnings.filterwarnings(action='once')
    sns.set()

    if not filename:
        filename = "resources/dataset_train.csv"
    data = read_csv(filename)
    data, column_list = prepare_dataframe(data, missing=False, norm=True)

    # Select columns with numeric values or not
    print("\nQUESTION: Quelles sont les deux features qui sont semblables ?")
    variables = column_list[6:19]

    # dessiner grande gride
    plt.rc('font', size=2)
    plt.rc('axes', labelsize=2)
    plt_scatters(data, variables, 1)
    plt.show()

    # repondre
    print("\nRESP1: Astronomy et Defense ont une repartition semblable") 
    print("mais correlée négativement")
    print("RESP2: Arithmancy et Magical Creatures ne sont pas correlees")
    print("mais elles ont toutes les 2 une repartition qui ressemble à aléatoire\n")

    # dessiner les petits plots
    plt.rc('font', size=12)
    plt.rc('axes', labelsize=12) 
    sns_scatter(data, 'Astronomy', 'Defense Against the Dark Arts', "Hogwarts House", 1)
    sns_scatter(data, 'Arithmancy', 'Care of Magical Creatures', "Hogwarts House", 2)
    plt.show()


main("")
