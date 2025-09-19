import pandas as pd
import numpy as np
import os
import scipy.stats
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

def normality(df, col, type, alpha, dir):
    #Reduce el df a solo el tipo de dato que se le esta pasando
    df = df.loc[df['Data_Type'] == type]

    #Generamos un histograma para cada muestra, se guarda en la carpeta 'img'
    #Esto es solo para visualizar como se ven las distribuciones
    #Para spoilear, las graficas de estimaciones tienen forma de campana
    #y en las de las proyecciones hay menos diferencia entre porcentajes
    histogram(df, col, type, dir)

    #Sacamos la estadistica de Shapiro-Wilk y el valor p
    #El stat solo esta para dar mas informacion
    stat, p = shapiro(df[col])

    if p > alpha:
        print(f"La distribucion de los {type} de la columna {col} es normal. \
              \nEstadistica de Shapiro-Wilk: {round(stat, 4)} \
              \nValor p: {p}")
        return True
    else:
        print(f"La distribucion los {type} de la columna {col} NO es normal. \
             \nEstadistica de Shapiro-Wilk: {round(stat, 4)} \
             \nValor p: {p}")
        return False

def anova(df, str_ols, col, alpha):
    modl = ols(str_ols, data=df).fit()
    #Se genera un dataframe con informacion de la prueba, incluyendo el valor p
    anova_df = sm.stats.anova_lm(modl, typ=2)

    print(f"\nPRUEBA DE ANOVA DE TIPOS DE DATO EN LA COLUMNA {col}\n")
    if anova_df["PR(>F)"][0] < alpha:
        print("Hay diferencias significativas entre los grupos.")
        print(anova_df)
        print("\n")
    else:
        print("No hay diferencias significativas entre los grupos.")
        print(anova_df)
        print("\n")

def dunn(df, col, alpha):
    #Creamos dataframes que solo tengan un tipo de dato 
    df_est = df[df['Data_Type'] == 'Estimate'][col]
    df_proj = df[df['Data_Type'] == 'Projection'][col]

    #Aplicamos la prueba de Kruskal-Wallis o Dunn en estas dos muestras
    stat, p = scipy.stats.kruskal(df_est, df_proj)

    print(f"\nPRUEBA DE KRUSKAL-WALLIS DE TIPOS DE DATO EN LA COLUMNA {col}\n")
    if p < alpha:
        print(f"Hay diferencias significativas entre los grupos. \
              \nEstadistica de Kruskal-Wallis: {round(stat, 4)} \
              \nValor p: {p}\n")
    else:
        print(f"No hay diferencias significativas entre los grupos. \
              \nEstadistica de Kruskal-Wallis: {round(stat, 4)} \
              \nValor p: {p}\n")

def histogram(df, col, type, dir):
    plt.figure(figsize=(10, 6))
    plt.hist(df[col], bins=10, edgecolor='black', alpha=0.7)

    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title(type)

    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(dir, f'../Practica 4/img/{col}-{type}-distribution.png'), bbox_inches='tight', dpi=300)

def main():
    direccion_actual = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(direccion_actual, '../Practica 2/births-and-deaths-statistics_(Year).csv'))

    #La observacion que hice en mi dataset, especialmente en la practica pasada, es que la distribucion de las 
    #diferencias interanuales de los datos son muy diferentes entre proyecciones y estimados
    #Para esto podemos checar las graficas en el folder de la practica pasada que terminan en yoy
    
    #Por lo tanto, la Hipotesis Nula (H0) es que no hay diferencias significativas entre las distribuciones
    #de las diferencias interanuales de nacimientos y muertes
    #La Hipotesis Alternativa (H1) es que si hay diferencias significativas

    #Calculamos las diferencias porcentuales interanuales
    df['Births_YoY'] = df['Births_Sum'].pct_change() * 100
    df['Deaths_YoY'] = df['Deaths_Sum'].pct_change() * 100

    #Eliminamos los valores NaN, en este caso el del primer año
    df = df.dropna()

    #Este dataset agrupado no tiene la columa de Data_Type, entonces se la agrego aqui
    df['Data_Type'] = np.where(df['Year'] <= 2023, 'Estimate', 'Projection')

    #Mismo alpha que en DataAnalysis.org en el repositorio
    alpha = 0.005
    
    columns = ['Births_YoY', 'Deaths_YoY']
    types = ['Estimate', 'Projection']

    #Para decidir si usar ANOVA o Dunn, primero hay que realizar una prueba de normalidad
    #Si tanto los estimados como las proyecciones son normales, usamos ANOVA y si no, usamos Dunn
    for col in columns:
        normals = 0
        
        for type in types:
            #Prueba de normalidad, regresa booleano
            is_normal = normality(df, col, type, alpha, direccion_actual)

            if is_normal:
                normals += 1
        
        #Si los estimados y proyecciones son normales va a ANOVA, en el caso
        #de este dataset ninguno de las 4 muestras son normales
        #Se hace una prueba para nacimientos y otra para muertes
        if normals == 2:
            #El modelo para el anova tiene el parametro formula
            #que es un string con forma de 'Variable dependiente' ~ 'Variable independiente'
            #En este caso estamos viendo si los datos de la columna son influidos por la categoria de dato
            #por lo que primero va la columna y luego Data_Type
            anova(df, f'{col} ~ Data_Type', col, alpha)
        else:
            dunn(df, col, alpha)

#Los resultados de la prueba es que hay diferencias significativas entre las estimados y las proyecciones
#Esto tiene sentido ya que es muy dificil que las proyecciones puedan replicar la volatilidad de los datos naturales
if __name__ == "__main__":
    main()