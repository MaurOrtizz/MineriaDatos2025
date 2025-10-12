import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from scipy.spatial.distance import cdist
from collections import Counter

def standardize(df, columns):
    #Normalizamos las columnas para que no haya problemas con el modelo
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
    return df
    
    #Tambien se puede realizar con una funcion
    '''
    scaler = StandardScaler()
    standard_array = scaler.fit_transform(df[columns])
    return pd.DataFrame(standard_array, columns=columns, index=df.index)
    '''

#Funcion para imprimir el scatter plot de las dos columnas que se usaron como atributos (normalizadas)
def scatter_plot(df, X, y, year):
    #Usamos solo los datos del año que se pasa a la funcion
    df = df[df["Year"] == year]

    direccion_actual = os.path.dirname(__file__)
    
    plt.figure(figsize=(10, 7))

    #Vamos agregando los datos de cada continente  
    for i, continent in enumerate(df[y].unique()):
        rows = df[df[y] == continent]
        plt.scatter(rows[X[0]], rows[X[1]], label=continent, s=80, alpha=1)
    
    plt.xlabel(f"Distribución de {X[0]} (estandarizado)")
    plt.ylabel(f"Distribución de {X[1]} (estandarizado)")
    plt.title(f"Distribución de paises en {year} por continente")

    plt.legend(title="Continente", loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(direccion_actual, f'../Practica 6/img/scatterplot{year}-{X}.png'), dpi=300)
    #plt.show()

#Funcion para imprimir matriz de confusion
def conf_matrix(y_test, y_pred, X):
    direccion_actual = os.path.dirname(__file__)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    
    plt.xlabel('Valores predecidos (Continentes)')
    plt.ylabel('Valores reales (Continentes)')
    plt.title(f'Matriz de Confusion - {X}')
    plt.tight_layout()

    plt.savefig(os.path.join(direccion_actual, f'../Practica 6/img/conf_matrix-{X}.png'), dpi=300)
    #plt.show()

#Impresion de puntaje del modelo por continente
def continent_report(y_test, predictions):
    report = classification_report(y_test, predictions, output_dict=True)
    
    print("\nPuntaje de precisión por continente:")
    for continent in np.unique(y_test):
        precision = report[continent]['precision']
        print(f"\t{continent}: {round(precision, 2)}")
    print("\n")

#Funcion para realizar el modelo de KNN usando la libreria sklearn
def knn(X, y, k, df):
    #Normalizamos las columnas
    df_estandar = standardize(df, X)
    
    #Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        df_estandar[X].values, df_estandar[y].values, test_size=0.3, random_state=42, stratify=df_estandar[y])
    
    #Creamos el modelo y lo ajustamos a los datos de entrenamiento
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    #Sacamos las predicciones para visualizarlas en la matriz de confusion
    predictions = knn.predict(X_test)
    #Sacamos el puntaje del modelo para ver su precisión
    score = knn.score(X_test, y_test)

    #Aqui puede cambiar el valor del año para ver los resultados de distintos años
    #En esta practica solo utilice valores de 2025 en el scatter plot porque las graficas con todos los datos
    #eran muy dificiles de leer por la cantidad de datos
    year = 2025
    scatter_plot(df, X, y, year)
    conf_matrix(np.array(y_test), np.array(predictions), X)

    print("=====MODELO DE CLASIFICACION K-NEAREST NEIGHBORS=====")
    print(f"Atributos del modelo: {X}")
    print(f"Etiquetas del modelo: Continentes")
    print(f"Puntaje de precisión: {round(score, 2)}")

    continent_report(y_test, predictions)

#Funcion para realizar el modelo de KNN sin usar la libreria sklearn
#Da resultados similares pero se tarda mas tiempo en llegar al resultado
def knn_manual(X, y, k, df):
    #Normalizamos las columnas
    df_estandar = standardize(df, X)
    
    #Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        df_estandar[X].values, df_estandar[y].values, test_size=0.3, random_state=42, stratify=df_estandar[y])

    #Calculamos la matriz de distancias euclideanas usando cdist
    #Esto tambien lo pude haber hecho con un ciclo for y con una funcion para
    #calcular las distancias euclideanas como en su repositorio pero
    #se tardaba mas de 6 minutos en dar resultados por la cantidad de datos
    #Es mas eficiente de esta forma 
    distances = cdist(X_test, X_train, metric='euclidean')

    #Sacamos los indices de los vecinos mas cercanos
    indices = np.argsort(distances, axis=1)[:, :k]

    #Realizamos las predicciones de los valores de y
    predictions = []
    for neighbors in indices:
        labels = y_train[neighbors]
        most_common = Counter(labels).most_common(1)[0][0]
        predictions.append(most_common)

    #Aqui puede cambiar el valor del año para ver los resultados de distintos años
    #En esta practica solo utilice valores de 2025 en el scatter plot porque las graficas con todos los datos
    #eran muy dificiles de leer por la cantidad de datos
    year = 2025
    scatter_plot(df, X, y, year)
    conf_matrix(np.array(y_test), np.array(predictions), X)
    
    #Sacamos el puntaje del modelo para ver su precisión
    score = np.mean(predictions == y_test)

    print("=====MODELO DE CLASIFICACION K-NEAREST NEIGHBORS=====")
    print(f"Atributos del modelo: {X}")
    print(f"Etiquetas del modelo: Continentes")
    print(f"Puntaje de precisión: {round(score, 2)}")

    continent_report(y_test, predictions)

def main():
    direccion_actual = os.path.dirname(__file__)
    
    #Agregue un dataset con informacion geografica para esta practica (voy a buscar clasificar los datos por continentes)
    #El codigo de carga y join con el dataset de la practica pasada se encuentra en DataLoading&Join.py
    #en la carpeta "new_dataset_scripts"
    df = pd.read_csv(os.path.join(direccion_actual, '../Practica 6/datasets/births-and-deaths_continents.csv'))

    #Calculo tasas de nacimiento y muerte como en la practica pasada
    df["Birth_Rate"] = df["Births_Combined"] / (df["Population in thousands"])
    df["Death_Rate"] = df["Deaths_Combined"] / (df["Population in thousands"])

    #Elimino las filas donde Entity es World, porque estas no tenian valores en el dataset con el que hice el join
    df = df[df['Entity'] != 'World']
    
    #Solo incluyo valores de paises porque no tiene sentido agregarlos en un modelo de clasificacion de datos por
    #continente y porque obviamente quedaron valores nulos con el dataset en esas filas
    df = df[df["Entity_Type"] == "Country"]    

    #En esta practica hice dos modelos de KNN, ambos los use para predecir el continente del que venia el dato
    #En un modelo uso las variables de poblacion urbana y tasa de dependencia
    #Y en otro la tasa de nacimientos y de muertes

    knn(X=["Urban Population (%)", "Dependency Ratio (%)"], y="region", k=5, df=df)
    #knn_manual(X=["Urban Population (%)", "Dependency Ratio (%)"], y="region", k=5, df=df)
    knn(X=["Birth_Rate", "Death_Rate"], y="region", k=5, df=df)
    #knn_manual(X=["Birth_Rate", "Death_Rate"], y="region", k=5, df=df)

    #INTERPRETACION DE LOS RESULTADOS:
        #El puntaje de ambos modelos me sorprendio bastante porque fue mucho mas bajo de lo que me esperaba,
        #pero supongo que dado que son 5 continentes y el valor salio alrededor de 0.45 y 0.5 significa que
        #el modelo es ligeramente mejor a predecir el continente que si solo adivinara
        #Tambien intente con otros valores de k pero los resultados fueron muy similares
        #Yo diria que esto se debe a que si bien estas columnas que utilice pueden dar informacion importante
        #en predecir de en que continente fue, hay varios factores que podrian ser mejores en predecir el continente
        #como el PIB per capita, solo que no he podido encontrar un dataset que tenga el pib per capita o medidas
        #economicas por pais y año.

    
if __name__ == '__main__':
    main()