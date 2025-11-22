import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

direccion_actual = os.path.dirname(__file__)

def regression_plot(train_df, test_df, x, y, continent, cutoff, pred_train, pred_test, pred_proj, pred_frame, proj_frame):
    direccion_actual = os.path.dirname(__file__)

    plt.figure(figsize=(12, 8))

    #Organizamos los dataframes en el eje x
    train_df = train_df.sort_values(by=x)
    test_df = test_df.sort_values(by=x)

    #Graficamos los puntos de entrenamiento (en este caso los estimados)
    plt.scatter(train_df[x], train_df[y], alpha=0.7, color="blue", label=f"Estimados (1950-{cutoff - 1})")
    #Graficamos la linea de regresion para los datos de entrenamiento
    plt.plot(train_df[x], pred_train, color="blue", label="Línea de Regresión")
    #Graficamos el intervalo de prediccion de los datos de entrenamiento
    plt.fill_between(train_df[x], pred_frame["obs_ci_lower"], pred_frame["obs_ci_upper"], color="blue",
                     label="Intervalo de Prediccion de predicciones", alpha=0.2)

    if len(test_df) > 0:
        #Graficamos los datos de prueba (en este caso las proyecciones)
        plt.scatter(test_df[x], test_df[y], alpha=0.7, color="orange", label=f"Predicciones ({cutoff}-2050))")
        #Continuamos la linea de regresion de los datos de entrenamiento, mostrando como es la tendencia
        #que predice el modelo de forecasting
        plt.plot(test_df[x], pred_test, color="blue", label="Línea de Tendencia del Forecasting")
        #Graficamos la linea de regresion de los datos de prueba
        plt.plot(test_df[x], pred_proj, color="red", label="Linea de Tendencia de Proyecciones")
        #Graficamos el intervalo de prediccion de los datos de prueba
        plt.fill_between(test_df[x], proj_frame["obs_ci_lower"], proj_frame["obs_ci_upper"],
                         color="orange", label="Intervalo de Prediccion de proyecciones", alpha=0.2)

    plt.title(f"Predicciones {y} - {continent}")
    plt.ylabel(y)
    plt.xlabel(x)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(direccion_actual, f'../PIA/forecasting/Forecasting-{y}-{continent}'), dpi=300)
    plt.close()

def regression(df, x, y, continent):
    #Aqui esta variable define la diferencia entre el dataset de entrenamiento
    #y el de prueba, para poder comparar el forecasting del modelo con los
    #datos reales. En este caso escogi 2024 porque es el año en el que empiezan
    #las proyecciones de datos.
    cutoff = 2024

    #Inicializamos los dataframes de entrenamiento y prueba
    train_df = df[(df["Year"] < cutoff) & (df["region"] == continent)]
    test_df = df[(df["Year"] >= cutoff) & (df["region"] == continent)]

    #Copiamos la columna X y le agregamos una constante, que es el intercepto
    x_constant = sm.add_constant(train_df[x])
    #Creamos y entrenamos el modelo
    model = sm.OLS(train_df[y], x_constant).fit()

    #Ya con el modelo entrenado le pedimos que prediga los valores de entrenamiento
    #Esto es solamente para agregar la linea de la regresion en la grafica 
    pred_train = model.predict(x_constant)
    #Igualmente sacamos el summary frame de las predicciones para graficar mas
    #facilmente el intervalo de prediccion
    pred_frame = model.get_prediction(x_constant).summary_frame(alpha=0.05)

    #Inicializamos error en 0 por si tomamos todos los datos como entrenamiento,
    #para que esta variable tenga valor
    error = 0

    if len(test_df) > 0:
        #Copiamos la columna X de los datos de prueba y le agregamos el intercepto,
        #esto nos va a ayudar a calcular el error del modelo y graficar tanto las
        #predicciones del modelo como los datos "reales" del dataset
        proj_const = sm.add_constant(test_df[x])
        #Creamos un modelo entrenadolo con los datos de prueba, este modelo nos va a servir
        #para graficar los datos "reales"
        proj_model = sm.OLS(test_df[y], proj_const).fit()
        #Le pedimos a este modelo que prediga los valores de prueba, que fue con los que fue entrenado
        #Esto es para graficar la linea de regresion de los datos de prueba
        pred_proj = proj_model.predict(proj_const)
        #Igualmente sacamos el summary frame de los datos de prueba para graficar
        #el intervalo de prediccion
        proj_frame = proj_model.get_prediction(proj_const).summary_frame(alpha=0.05)

        #Le pedimos a nuestro modelo original que prediga los datos de prueba
        #Asi en la grafica vamos a poder visualizar que tan cercano estuvo a los datos "reales"
        pred_test = model.predict(proj_const)
        #Calculamos el error para que se imprima
        error = np.mean((pred_test - test_df[y].values)**2)

    #Funcion de graficar el forecasting
    regression_plot(train_df, test_df, x, y, continent, cutoff, pred_train, pred_test, pred_proj, pred_frame, proj_frame)

    print("=====REGRESION LINEAL=====")
    print(f"Variable dependiente (y): {y}")
    print(f"Continente: {continent}")
    print(f"Error de predicción de los datos de 2024-2050: {round(error, 2) if error != 0 else 'No aplica'}")
    print("\n")
    print(model.summary())
    print("\n")

def forecasting(df):
    #Agrupamos por continente para la regresion
    df_bycontinent = df.groupby(["region", "Year", "Data_Type"], as_index=False).agg({
        "Births_Combined": "sum", "Deaths_Combined": "sum", "Population in thousands": "sum",
        "Urban Population (%)": "mean", "Dependency Ratio (%)": "mean"
    })

    df_bycontinent["Birth_Rate"] = df_bycontinent["Births_Combined"] / (df_bycontinent["Population in thousands"])
    df_bycontinent["Death_Rate"] = df_bycontinent["Deaths_Combined"] / (df_bycontinent["Population in thousands"])

    #Realizamos el forecasting de la tasa de nacimientos y muertes para cada continente
    for continent in df_bycontinent["region"].unique():
        regression(df_bycontinent, x="Year", y="Birth_Rate", continent=continent)
        regression(df_bycontinent, x="Year", y="Death_Rate", continent=continent)

def outliers(df, pca_fit):
    #Agregamos los valores de PC1 y PC2 al dataframe original
    df['PC1'] = pca_fit[:, 0]
    df['PC2'] = pca_fit[:, 1]
    #Calculamos la distancia del origen hasta el punto
    df['distance'] = np.sqrt(df['PC1']**2 + df['PC2']**2)

    #Ordenamos de mayor a menos
    df_outlier = df.sort_values('distance', ascending=False)

    #Aqui tome 11 outliers porque en la grafica de los KMeans se pueden notar 11 puntos
    #que estan mas alejados de la gradiente principal
    N = 11
    df_outlier.head(N).to_csv(os.path.join(direccion_actual, "outliers/outliers.csv"), index=False)

def DBSCAN_clusters(df, pca_fit):
    #DBSCAN es un algoritmo de clustering que agrupa clusters basado en la densidad de los puntos
    #Aqui lo uso para corrobar mi interpretacion de que no hay clusters naturales en el dataframe
    
    #Inicializamos el modelo DBSCAN con el PCA ajustado
    db = DBSCAN(eps=0.30, min_samples=10).fit(pca_fit)
    #Guardamos las etiquetas
    df['DBSCAN'] = db.labels_
    #Calculamos la cantidad de clusters que se encontraron
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"Clusters encontrados con DBSCAN: {n_clusters}")

def kmeans(df, columnas_numericas):
    #Agarramos las columnas numericas del dataset que van a ser la base
    #de los componentes principales de KMeans
    X_year = df[columnas_numericas].to_numpy()

    #Estandarizamos el dataframe
    scaler = StandardScaler()
    X_year_scaled = scaler.fit_transform(X_year)

    #Inicializamos PCA a 2 componentes que contienen informacion de las 4 variables numericas
    pca_year = PCA(n_components=2)
    #Ajustamos el PCA con los datos del dataframe
    pca_fit = pca_year.fit_transform(X_year_scaled)

    #Imprimimos la composicion de los componentes principales
    pca_components = pd.DataFrame(pca_year.components_, columns=columnas_numericas, index=['PC1', 'PC2'])
    pca_components.to_csv(os.path.join(direccion_actual, "outliers/pca_components.csv"))

    #Iteramos KMeans con diferentes cantidades de clusters para ver como
    #cambia el silhouette score 
    for k in [2, 3, 4, 5]:
        #Definimos el Modelo de KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pca_fit)
        df[f"cluster_k{k}"] = labels

        plt.figure(figsize=(10, 7))
        plt.scatter(pca_fit[:, 0], pca_fit[:, 1], c=labels, cmap='viridis', s=8, alpha=0.5)
        plt.title(f"KMeans - k = {k}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid()
        plt.colorbar()
        plt.savefig(os.path.join(direccion_actual, f"kmeans/kmeans_{k}.png"))
        plt.close()

        print(f"Silhouette score para k={k}: {round(silhouette_score(pca_fit, labels), 4)}")

    #Analisis de outliers
    outliers(df, pca_fit)

    #Corroboracion de resultados de KMeans con DBSCAN
    DBSCAN_clusters(df, pca_fit)


def asia_timeseries(df):
    #Agarramos solo los datos de Asia Oriental
    df_asia = df[df["sub-region"] == "Eastern Asia"]

    #Graficamos la tasa de natalidad y mortalidad para todos los años de todos los paises
    #de Asia Oriental para verificar como se comparan los datos de Japon
    for column in ["Birth_Rate", "Death_Rate"]:
        plt.figure(figsize=(12, 6))
        for entity in df_asia["Entity"].unique():
            country = df_asia[df_asia["Entity"] == entity]
            if entity == "Japan":
                plt.plot(country["Year"], country[column], label=entity, linewidth=3, alpha=1.0)
            else:
                plt.plot(country["Year"], country[column], label=entity)
        plt.title(f"{column} - Asia Oriental")
        plt.xlabel("Year")
        plt.ylabel(column)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(direccion_actual, f"Eastern_Asia/EastAsia_{column}.png"))
        plt.close()

def main():
    direccion_actual = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(direccion_actual, '../Practica 6/datasets/births-and-deaths_continents.csv'))

    #Calculamos tasas de nacimientos y muertes
    df['Birth_Rate'] = (df['Births_Combined'] / df['Population in thousands'])
    df['Death_Rate'] = (df['Deaths_Combined'] / df['Population in thousands'])

    columnas_numericas = ['Birth_Rate', 'Death_Rate', 'Urban Population (%)', 'Dependency Ratio (%)']

    #Creamos carpetas donde van a ir las graficas
    os.makedirs(os.path.join(direccion_actual, "forecasting"), exist_ok=True)
    os.makedirs(os.path.join(direccion_actual, "outliers"), exist_ok=True)
    os.makedirs(os.path.join(direccion_actual, "kmeans"), exist_ok=True)
    os.makedirs(os.path.join(direccion_actual, "Eastern_Asia"), exist_ok=True)

    #Ejecutamos los analisis

    #Analisis de KMeans con componentes principales y analisis de outliers
    kmeans(df, columnas_numericas)

    #Comparacion de tasas de natalidad y mortalidad en Asia Oriental durante los años
    asia_timeseries(df)

    #Regresion Lineal de Forecasting para tasa de natalidad y mortalidad por continente
    forecasting(df)

if __name__ == "__main__":
    main()