import json
import os
import numpy as np
from scipy.stats import f_oneway, levene, f
from statistics import pstdev, pvariance

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def perform_anova(*groups):
    f_statistic, p_value = f_oneway(*groups)
    return f_statistic, p_value

def perform_levene(*groups):
    stat, p_value = levene(*groups)
    return stat, p_value

def get_pstdev(population):
    return pstdev(population)

def get_pvar(population):
    return pvariance(population)
