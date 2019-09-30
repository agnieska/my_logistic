
"""
Faites un script nommé scatter_plot.[extension] qui affiche un scatter plot répondant à la question suivante : 
Quelles sont les deux features qui sont semblables ?
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import warnings; warnings.filterwarnings(action='once')
import numpy as np

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

def plt_scatter(df, col_name1, col_name2, fig_nb):
    plt.rc('font', size=12)
    plt.rc('axes', labelsize=12) 
    plt.figure(fig_nb)
    ax = plt.axes()
    ax.scatter(x=df[col_name1], y=df[col_name2], color=np.random.rand(3), s=3)
    ax.set_title(col_name1+" by "+col_name2)
    plt.xlabel(col_name1[:12])
    plt.ylabel(col_name2[:12])
    #plt.show()

def sns_scatter(df, col_name1, col_name2, col_name3, fig_nb):
    plt.rc('font', size=12)
    plt.rc('axes', labelsize=12) 
    plt.figure(fig_nb)
    ax = sns.scatterplot(data=df, x=col_name1, y=col_name2, hue=col_name3, size=3)
    ax.set_title("SNS"+col_name1+" by "+col_name2)
    #plt.xlabel(col_name1[:12])
    #plt.ylabel(col_name2[:12])
    #plt.show()

def plt_scatters(df, variables, fig_nb):
    n_rows = len(variables)
    n_cols = len(variables)
    plt.rc('font', size=2)
    plt.rc('axes', labelsize=10) 
    fig = plt.figure(fig_nb, figsize=(10,10), dpi= 100)
    for i, var_name1 in enumerate(variables):  
        for j, var_name2 in enumerate(variables):          
            #fig = plt.figure()
            #ax = plt.axes()
            #number = (i+1)*(j+1)
            number = i*len(variables)+j+1
            ax=fig.add_subplot(n_rows, n_cols, number)
            ax.scatter(x=df[var_name1], y=df[var_name2], color=np.random.rand(3), s=0.5)
            
            if (number-1) % n_cols==0:
                plt.ylabel(var_name1[:6])
            if number > (n_rows-1)* n_cols:
                plt.xlabel(var_name2[:6])
            
            #plt.title('Training data ')
            #plt.show()
    fig.tight_layout()  # Improves appearance a bit.
    #plt.show()

def sns_scatters(df, variables, variable3, fig_nb):
    n_rows = len(variables)
    n_cols = len(variables)
    plt.rc('font', size=2)
    plt.rc('axes', labelsize=10) 
    fig = plt.figure(fig_nb, figsize=(10,10), dpi= 100)
    for i, var_name1 in enumerate(variables):  
        for j, var_name2 in enumerate(variables):          
            number = i*len(variables)+j+1
            ax = fig.add_subplot(n_rows, n_cols, number)
            ax = sns.scatterplot(data=df, x=var_name1, y=var_name2, hue=variable3, size=0.5)
            #ax.scatter(x=df[var_name1], y=df[var_name2], color=np.random.rand(3), s=1)
            if (number-1) % n_cols==0:
                plt.ylabel(var_name1[:6])
            if number > (n_rows-1)* n_cols:
                plt.xlabel(var_name2[:6])
    ax.set_title("Variables by "+variable3)
    fig.tight_layout()  # Improves appearance a bit.
    #plt.show()

def main ():

    data, column_list = prepare_dataframe ("resources/dataset_train.csv")
    # Select columns with numeric values or not
    """ variables = column_list[6:19] """
    
    # dessiner grande gride
    """ plt.rc('font', size=2)
    plt.rc('axes', labelsize=2) 
    plt_scatters(data, variables, 1)
    sns_scatters(data, variables, "Hogwarts House", 1)
    plt.show() """
   
   # dessiner les petits plots
    plt.rc('font', size=12)
    plt.rc('axes', labelsize=12) 
    plt_scatter(data, 'Potions', 'Care of Magical Creatures', 1)
    plt_scatter(data, 'Arithmancy', 'Care of Magical Creatures', 2)
    sns_scatter(data, 'Potions', 'Care of Magical Creatures', "Hogwarts House", 3)
    sns_scatter(data, 'Arithmancy', 'Care of Magical Creatures', "Hogwarts House", 4)
    plt.show()

    

main()