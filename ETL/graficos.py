#%%
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator

import warnings
warnings.filterwarnings("ignore") 

import scipy.stats as stats
from scipy.stats import shapiro, poisson, chisquare, expon, kstest
from scipy.stats import levene, bartlett, shapiro

import pymysql
from sqlalchemy import create_engine
#%%
def histograma(df1):
    columnas_number_con_nulos = df1[df.columns[df1.isnull().any()]].select_dtypes(include = np.number).columns
    for col in list(columnas_number_con_nulos):
        plt.figure(figsize=(8, 5))
        plt.hist(df1[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.show()
#%%

colores_proyecto = [
    'mediumpurple',   # morado medio
    'darkorange',     # naranja fuerte
    'plum',           # morado claro
    'sandybrown',     # naranja suave
    'purple',         # morado intenso
    'lightsalmon',    # naranja rosado
    'orchid',         # morado rosado
    'chocolate'       # naranja oscuro
]