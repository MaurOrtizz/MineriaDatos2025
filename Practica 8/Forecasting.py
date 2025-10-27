import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd
import os

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
    scatter_plot(train_df, test_df, x, y, continent, cutoff, pred_train, pred_test, pred_proj, pred_frame, proj_frame)

    print("=====REGRESION LINEAL=====")
    print(f"Variable dependiente (y): {y}")
    print(f"Continente: {continent}")
    print(f"Error de predicción de los datos de 2024-2050: {round(error, 2) if error != 0 else "No aplica"}")
    
    print("\n")
    print(model.summary())
    print("\n")

def scatter_plot(train_df, test_df, x, y, continent, cutoff, pred_train, pred_test, pred_proj, pred_frame, proj_frame):
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
    
    plt.savefig(os.path.join(direccion_actual, f'../Practica 8/img/Forecasting-{y}-{continent}'), dpi=300)
    #plt.show()

def main():
    direccion_actual = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(direccion_actual, '../Practica 6/datasets/births-and-deaths_continents.csv'))

    #Agrupamos por continente para tener datos historicos de cada continente 
    df_bycontinent = df.groupby(["region", "Year", "Data_Type"], as_index=False).agg({
        "Births_Combined": "sum", "Deaths_Combined": "sum", "Population in thousands": "sum", 
        "Urban Population (%)": "mean", "Dependency Ratio (%)": "mean"})
    
    #Calculamos la tasa de nacimientos y muertes
    df_bycontinent["Birth_Rate"] = df_bycontinent["Births_Combined"] / (df_bycontinent["Population in thousands"])
    df_bycontinent["Death_Rate"] = df_bycontinent["Deaths_Combined"] / (df_bycontinent["Population in thousands"])

    #Para cada continente vamos a hacer un modelo de forecasting para Birth_Rate y Death_Rate
    for continent in df_bycontinent["region"].unique():
        regression(df_bycontinent, x="Year", y="Birth_Rate", continent=continent)
        regression(df_bycontinent, x="Year", y="Death_Rate", continent=continent)

    #Interpretacion de los datos:
        #En general el modelo fue mas acertado en predecir la tasa de nacimientos, calculando datos un poco mas bajos
        #de los que estiman las predicciones, a excepcion de Africa. Igualmente tampoco se puede confiar completamente
        #en los datos del dataset, debido a que son predicciones y en lo personal, siento que el modelo es mas acertado
        #con la realidad a futuro. 
        #Igualmente hubo una diferencia mas grande entre los datos predichos para la tasa de muertes y las proyecciones.
        #Viendo las graficas para cada continente, puede que un modelo de regresion de otro tipo que no sea lineal pudo
        #haber generado predicciones mas acertadas.

if __name__ == '__main__':
    main()