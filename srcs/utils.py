import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def dual_likelihood(category_num, dataList, previous_mean, current_mean, data_var, total_category):
    current_likelihood = 0
    previous_likelihood = 0
    #Use left truncated normal truncated at 0 for category 0, since the minimum score is 0
    if category_num == 0:
        current_likelihood = sum(stats.truncnorm.logpdf(
            dataList, 
            a = ((0-current_mean)/(data_var)), 
            b = np.inf,
            loc = current_mean, 
            scale = data_var))
        previous_likelihood = sum(stats.truncnorm.logpdf(
            dataList, 
            a = ((0-previous_mean)/(data_var)), 
            b = np.inf, 
            loc = previous_mean, 
            scale = data_var))
    #Use right truncated normal truncated at 100.01 for the last category, since the maximum score is 100. 
    #added 0.01 for inclusion of score 100.
    elif category_num == total_category - 1:
        current_likelihood = sum(stats.truncnorm.logpdf(
            dataList, 
            a = -np.inf, 
            b = ((100.01-current_mean)/(data_var)),
            loc = current_mean, 
            scale = data_var))
        previous_likelihood = sum(stats.truncnorm.logpdf(
            dataList, 
            a = -np.inf, 
            b = ((100.01-previous_mean)/(data_var)), 
            loc = previous_mean, 
            scale = data_var))
    #categories in between don't need truncation. Use normal distributions.
    else:
        current_likelihood = sum(stats.norm.logpdf(
            dataList, 
            loc = current_mean, 
            scale = data_var))
        previous_likelihood = sum(stats.norm.logpdf(
            dataList, 
            loc = previous_mean, 
            scale = data_var))
    return current_likelihood, previous_likelihood

def single_likelihood(category_num, dataList, mean_parameter, data_var, num_categories):
    likelihood = 0
    #Use left truncated normal truncated at 0 for category 0, since the minimum score is 0
    if category_num == 0: 
        likelihood = sum(stats.truncnorm.logpdf(
            dataList, 
            a = ((0-mean_parameter)/(data_var)), 
            b = np.inf, 
            loc = mean_parameter, 
            scale = data_var))
    #Use right truncated normal truncated at 100.01 for the last category, since the maximum score is 100. 
    #added 0.01 for inclusion of score 100.
    elif category_num == num_categories-1:
        likelihood = sum(stats.truncnorm.logpdf(
            dataList, 
            a = -np.inf, 
            b = ((100-mean_parameter)/(data_var)), 
            loc = mean_parameter, 
            scale = data_var))
    #categories in between don't need truncation. Use normal distributions.
    else:
        likelihood = sum(stats.norm.logpdf(
            dataList, 
            loc = mean_parameter, 
            scale = data_var))
    return likelihood