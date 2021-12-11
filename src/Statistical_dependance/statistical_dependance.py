# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt
import pandas as pd

# Description of this script:
# allow the user to test some statistical dependance between different features

def stat_dep_cat_cont(conf, data, var_cat, var_cont):
    """
    Test the statistical dependance between a continuous feature and a categorical feature of a dataset
    It returns a graph and an object with the result of the statistical test

    Args:
        conf : conf file
        data : The dataset
        var_cat : the categorical variable
        var_cont : the continuous variable

    Returns:
        the result of the test
    """

    # Do an Anoa test
    model = ols(var_cont + ' ~ ' + var_cat, data=data).fit()
    result_test = sm.stats.anova_lm(model, typ=2)
    # Create a visual graph 
    plt.figure(figsize=(12,8)) 
    sns.boxplot(y=var_cont, x=var_cat, data=data)
    plt.savefig(conf['Outputs_path'] + conf['folder_stats']
              + conf['selected_dataset'] + "_" + var_cat + "_" + var_cont +
              '_dependance.png')

    return result_test

def stat_dep_cont_cont(conf, data, var1, var2):
    """
    Test the statistical dependance between 2 continuous features of a dataset
    It returns a graph and an object with the result of the statistical test

    Args:
        conf : conf file
        data : The dataset
        var1 : the 1st variable
        var2 : the 2nd variable

    Returns:
        the result of the test
    """
    # pearson test need 
    df = data.dropna(subset=[var1, var2])

    # Do a pearson test 
    result_test = pearsonr(df[var1], df[var2])
    # Create a visual graph 
    plt.figure(figsize=(12,8)) 
    sns.regplot(x=var1, y=var2, fit_reg=True, data=df)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.savefig(conf['Outputs_path'] + conf['folder_stats']
              + conf['selected_dataset'] + "_" + var1 + "_" + var2 +
              '_dependance.png')

    return result_test

def stat_dep_cat_cat(conf, data, var1, var2, cmap="YlGnBu"):
    """
    Test the statistical dependance between 2 continuous features of a dataset
    It returns a graph and an object with the result of the statistical test

    Args:
        conf : conf file
        data : The dataset
        var1 : the 1st variable
        var2 : the 2nd variable
        cmap : Defaults to "YlGnBu".

    Returns:
        the result of the test
    """

    # Create a cross table
    contigency= pd.crosstab(data[var1], data[var2])
    # Do a Chi 2 Test
    result_test = chi2_contingency(contigency) 
    # Create a visual graph 
    plt.figure(figsize=(12,8)) 
    sns.heatmap(contigency, annot=True, cmap=cmap)
    plt.savefig(conf['Outputs_path'] + conf['folder_stats']
              + conf['selected_dataset'] + "_" + var1 + "_" + var2 +
              '_dependance.png')

    return result_test
