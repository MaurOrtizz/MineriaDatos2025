import kaggle
import os
import pandas as pd

#Cargamos datos desde Kaggle
def kaggle_dataset():
    kaggle.api.authenticate()

    kaggle.api.dataset_download_file(
        dataset = 'lucafrance/the-world-factbook-by-cia',
        file_name = 'countries.csv',
        path = './Practica 9/datasets',  #Estoy ejecutando el script desde el directorio de todas las practicas
        force=True)

#Al igual que con otras practicas, muchos paises tienen nombres distintos en este dataset
#entonces tuve que cambiarles el nombre para que concuerden y se pudiera hacer el Join
def country_dict():
    #Varios paises en este nuevo dataset tienen nombres distintos a los que vienen el viejo dataset
    #entonces el diccionario esta para cambiar los nombres de estos paises en este dataset
    #para que coincidan con el dataset original
    mapping = {
        "Bahamas, The" : "Bahamas",
        "Congo, Republic of the" : "Congo",
        "Falkland Islands (Islas Malvinas)" : "Falkland Islands",
        "Gambia, The" : "Gambia",
        "Macau" : "Macao",
        "Burma" : "Myanmar",
        "Saint Helena, Ascension, and Tristan da Cunha" : "Saint Helena",
        "Saint Martin" : "Saint Martin (French part)",
        "Turkey (Turkiye)" : "Turkey",
        "Virgin Islands" : "United States Virgin Islands",
    }

    return mapping

def drop_rows(df):
    #Al probar el join del dataset nuevo con el viejo descubri que el nuevo dataset no tiene valores para
    #Costa de Marfil, Sahara Occidental y los territorios franceses de ultramar, ademas de ningun territorio
    #Por eso, los voy a tener que eliminar del dataset viejo para que no haya nulls en la columna de Background
    #en el dataset junto.
    no_data = ["Cote d'Ivoire", "French Guiana", "Guadeloupe", "Martinique", "Mayotte", "Reunion", "Western Sahara"]
    df = df[(~df["Entity"].isin(no_data)) & (df["Entity_Type"] == "Country")]

    return df

def main():
    kaggle_dataset()

    direccion_actual = os.path.dirname(__file__)
    df_original = pd.read_csv(os.path.join(direccion_actual, '../../Practica 6/datasets/births-and-deaths_continents.csv'))
    df_new = pd.read_csv(os.path.join(direccion_actual, '../../Practica 9/datasets/countries.csv'))

    #Elimino registros que no tienen datos en el nuevo dataset en la columna de Background, 
    #entro mas a detalle dentro de la funcion
    df_original = drop_rows(df_original)
    df_original = df_original.groupby("Entity", as_index=False).agg({
        "Births_Combined": "sum", "Deaths_Combined": "sum", "Population in thousands": "sum", 
        "Urban Population (%)": "mean", "Dependency Ratio (%)": "mean", "region" : "first"
    })

    #Renombro la columna con los nombres de los paises para que no quede una columna duplicada despues del Join
    #Igualmente renombro la columna de Introduccion para que tenga un nombre mas corto
    df_new = df_new.rename(columns={"Country" : "Entity", "Introduction: Background" : "Background"})

    #Agarro solo estas columnas del dataset
    #El dataset tiene casi 1000 columnas, pero para la practica de Word Cloud creo que usar la de Background es mas
    #que suficiente, ya que tiene varias lineas por pais describiendo como es y un poco de su historia
    df_new = df_new[["Entity", "Background"]]

    #Realizo el mapeo de nombres de paises
    df_new["Entity"] = df_new["Entity"].replace(country_dict())

    join_df = pd.merge(df_original, df_new, left_on=["Entity"], right_on=["Entity"], how="left")

    join_df.to_csv(os.path.join(direccion_actual, '../datasets/births-and-deaths_descriptions.csv'), index=False)

if __name__ == "__main__":
    main()