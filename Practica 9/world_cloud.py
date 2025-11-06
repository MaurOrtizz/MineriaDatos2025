import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os

direccion_actual = os.path.dirname(__file__)

def create_cloud(df, continent=None):
    #Juntamos todas las descripciones de los paises del dataframe que se paso
    all_words = " ".join(df["Background"].astype(str))

    #Creamos el word cloud
    wordcloud = WordCloud(
        width=1200, height=600, background_color="white", min_font_size=5
    ).generate(all_words)

    plt.figure(figsize=(10, 5), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    #Guardamos las imagenes, el wordcloud global solo se guarda como wordcloud.png
    #y para el resto se especifica el continente
    if continent:
        plt.savefig(os.path.join(direccion_actual, f'../Practica 9/img/wordcloud_{continent}.png'))
    else:
        plt.savefig(os.path.join(direccion_actual, '../Practica 9/img/wordcloud.png'))
    plt.close()

def main():
    direccion_actual = os.path.dirname(__file__)
    #Este nuevo csv es resultado de un join que hice con un dataset
    #que tiene una descripcion para cada pais, lo cual es necesario para hacer la actividad
    #El codigo y documentacion de este dataset viene en Practica 9/dataset_scripts/DatasetLoading.py
    df = pd.read_csv(os.path.join(direccion_actual, '../Practica 9/datasets/births-and-deaths_descriptions.csv'))

    #Creamos la carpeta de imagenes
    os.makedirs(os.path.join(direccion_actual, '../Practica 9/img/'), exist_ok=True)

    #Vamos a crear un word cloud para cada continente, ademas de uno global
    continents = ["Americas", "Europe", "Asia", "Africa", "Oceania"]
    create_cloud(df)
    for continent in continents:
        #Solo tomamos datos para el continente actual
        df_continent = df[df["region"] == continent]
        create_cloud(df_continent, continent=continent)

if __name__ == "__main__":
    main()
