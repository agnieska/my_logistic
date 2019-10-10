"""
Faites un script nommé histogram qui affiche un histogram répondant 
à la question suivante : Quel cours de Poudlard a une répartition des notes homogènes entre les quatres maisons ?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from my_library import read_csv, prepare_dataframe


def plt_histograms(fig_nb, df, variables, n_cols=3):
    n_rows = (len(variables) / n_cols) + 1
    fig = plt.figure(fig_nb, figsize=(10, 10), dpi=100)
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        df[var_name].hist(bins=10, ax=ax, color=np.random.rand(3))
        ax.set_title(var_name[:12]+".  -distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()


def sea_hist(fig_nb, data, column_name):
    plt.figure(fig_nb, figsize=(8, 5), dpi=100)
    sns.distplot(
        data.loc[data['Hogwarts House'] == 'Ravenclaw', column_name],
        color="dodgerblue",
        label="Ravenclaw",
        hist_kws={'alpha': 0.7},
        kde_kws={'linewidth': 2}
        )
    sns.distplot(
        data.loc[data['Hogwarts House'] == 'Slytherin', column_name],
        color="orange",
        label="Slytherin",
        hist_kws={'alpha': 0.7},
        kde_kws={'linewidth': 2}
        )
    sns.distplot(
        data.loc[data['Hogwarts House'] == 'Gryffindor', column_name],
        color="green",
        label="Gryffindor",
        hist_kws={'alpha': 0.7},
        kde_kws={'linewidth': 2}
        )
    sns.distplot(
        data.loc[data['Hogwarts House'] == 'Hufflepuff', column_name],
        color="orchid",
        label="Hufflepuff",
        hist_kws={'alpha': 0.7},
        kde_kws={'linewidth': 2}
        )
    plt.ylim(0, 6)
    plt.title(column_name[:12]+'.  / by House name', fontsize=18)
    plt.legend()
    # plt.show()


def main(filename):
    warnings.filterwarnings(action='once')

    print("\nQUESTION: Quel cours de Poudlard a une répartition des notes") 
    print("homogènes entre les quatres maisons ?")
    
    if not filename:
        filename = "resources/dataset_train.csv"
    data = read_csv(filename)
    data, column_list = prepare_dataframe(data, missing=True, norm=True)

    # Select columns with numeric values
    variables = column_list[6:19]
    
    # Plot figures
    plt_histograms(1, data, variables, 3)
    print("\nRESPONSE: Care of Magical Creatures and Arithmancy")
    print("ont une répartition des notes homogènes entre les quatres maisons\n")
    sea_hist(2, data, 'Care of Magical Creatures')
    sea_hist(3, data, 'Arithmancy')
    sea_hist(4, data, 'Potions')
    plt.show()

main("")
