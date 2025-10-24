import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def standardize(df, columns):
    #Normalizamos las columnas para que no haya problemas con el modelo
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        
        if std == 0:
            df.loc[:, col] = 0.0
        else:
            df.loc[:, col] = (df[col] - mean) / std
    return df[columns]
    
    #Tambien se puede realizar con una funcion
    '''
    scaler = StandardScaler()
    standard_array = scaler.fit_transform(df[columns])
    return pd.DataFrame(standard_array, columns=columns, index=df.index)
    '''

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p2 - p1) ** 2))

#Funcion para recalcular las medias del modelo
def calculate_means(array, labels, k):
    means = []
    
    for cluster in range(k):
        #Creamos un arreglo de puntos del centroide k
        points = array[labels == cluster]
        #Recalculamos la media si el centroide no esta vacio
        if len(points) > 0:
            mean = np.mean(points, axis=0)
        else:
            mean = array[np.random.choice(len(array))]
        #Agregamos la media a la lista
        means.append(mean)
    
    #Convertimos la lista de medias a un arreglo
    means = np.array(means)
        
    return means

def scatter_plot(df_std, labels, centroids, k, iteration, year, x_col, y_col, x_idx, y_idx):

    #Graficamos cada cluster
    for cluster in range(k):
        #Solo tomamos los puntos de ese cluster
        points = df_std[labels == cluster]
        #Graficamos cada punto tomando en cuenta sus posiciones en "x" y "Y"
        plt.scatter(points[:, x_idx], points[:, y_idx], label=f"Cluster {cluster + 1}", s=25)
    
    #Graficamos la posicion de los centroides
    plt.scatter(centroids[:, x_idx], centroids[:, y_idx], marker=".", s=100, c="black", linewidths=1.2, label="Centroides")
    
    plt.title(f"K-Means de Valores de {year}: Iteración {iteration}")
    plt.xlabel(f"{x_col} (Estandarizado)")
    plt.ylabel(f"{y_col} (Estandarizado)")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"Practica 7/img_{year}/kmeans_{year}_{iteration + 1}.png", dpi=150)
    plt.close()

def kmeans(df, k, year, max_iterations):
    #Lista de columnas que se van a usar en el modelo: Year es para filtrar por año y las demas son los features
    columns = ["Year", "Birth_Rate", "Death_Rate", "Population in thousands", "Urban Population (%)", "Dependency Ratio (%)"]
    
    df = df[df["Year"] == year]
    
    df_std = standardize(df, columns)

    os.makedirs(f"Practica 7/img_{year}", exist_ok=True)

    #Droppeamos Year para que no afecte al modelo
    df_std = df_std.drop(columns="Year")

    #Pasamos a arreglo para que se hagan más rapido los calculos
    array_std = np.array(df_std)

    #Inicializamos el modelo agarrando los valores de k filas aleatorias
    rand_cent = np.random.choice(len(array_std), size=k, replace=False)

    #Sacamos los valores de esas k filas
    centroids = array_std[rand_cent]

    print(f"\n====Algoritmo K-Means para datos del año {year}====\n")

    #Columnas para la funcion de grafica
    x_col = "Birth_Rate"
    y_col = "Death_Rate"
    #Indices de la columnas "x" y "y" de la grafica
    x_idx = df_std.columns.get_loc(x_col)
    y_idx = df_std.columns.get_loc(y_col)

    for i in range(max_iterations):
        old_cent = centroids.copy()
        
        #Para cada fila (pais) en el array, se calcula la distancia euclideana de ese punto a cada centroide
        #y se escoge la distancia menor, y el resultado de todas las filas se pasa a un arreglo
        labels = np.array([np.argmin([euclidean_distance(cent, value) for cent in centroids]) for value in array_std])
        
        #Recalculamos los centroides
        means = calculate_means(array_std, labels, k)

        #Reasignamos los centroides a su nuevo valor
        centroids = means

        #Creamos la grafica para la iteracion
        scatter_plot(array_std, labels, centroids, k, i, year, x_col=x_col, y_col=y_col, x_idx=x_idx, y_idx=y_idx)

        #Checamos si el modelo convergio o no
        if (np.all(centroids == old_cent)):
            print(f"Convergencia alcanzada en la iteración {i + 1}\n")
            break
        if (i == max_iterations - 1):
            print("Modelo no convergio")

def main():
    direccion_actual = os.path.dirname(__file__)
    
    df = pd.read_csv(os.path.join(direccion_actual, '../Practica 6/datasets/births-and-deaths_continents.csv'))

    #Calculamos tasa de nacimientos y muertes
    df["Birth_Rate"] = df["Births_Combined"] / (df["Population in thousands"])
    df["Death_Rate"] = df["Deaths_Combined"] / (df["Population in thousands"])

    #Lo que se me hizo mejor para la practica fue hacer el algoritmo kmeans con datos de un año en particular,
    #para ir viendo como van cambiando los datos. Escogi 1950 porque es el primer año en el dataset, 2023 porque
    #es el ultimo año con datos "reales" y 2050 porque es el ultimo año en el dataset
    kmeans(df, k=3, year=1950, max_iterations=20)
    kmeans(df, k=3, year=2023, max_iterations=20)
    kmeans(df, k=3, year=2050, max_iterations=20)

    #Interpretacion de los datos:
    #Algo que notamos en los datos de 1950 es que hay mas diversidad en los puntos: un cluster tiene baja
    #natalidad y baja mortalidad (posiblemente representando paises ricos), otro tiene los datos promedios de
    #natalidad y relativamente baja mortalidad (paises en vias de desarrollo) y finalmente, 
    #hay un cluster de alta mortalidad/natalidad (paises pobres).
    #En 2023 vemos que los datos se alinean mas a tener una tasa de mortalidad mas baja, siendo la mayor diferencia
    #entre los tres clusters su tasa de natalidad (y posiblemente otras diferencias en columnas que no estan en la grafica)
    #Yo diria que esto es por avances en la medicina que han reducido drasticamente el indice de mortalidad en paises subdesarrollados
    #Finalmente, para el modelo de los datos proyectados de 2050, vemos que no es muy diferente que la distribicion
    #que se encuentra en los datos de 2023, con la diferencia que ahora todos los datos tienden a un indice de mortalidad mas alto.
    #Para los datos del 2023 y 2050, es claro que los clusters representan 3 niveles economicos de paises 
    #(desarrollado, en vias de desarrollo y subdesarrollado). Lo que mas me sorprendio es ver el rango mas grande que habia en
    #la tasa de mortalidad de 2050 y como disminuyo bastante en 2023. 

if __name__ == "__main__":
    main()