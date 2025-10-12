import kaggle
import os
import pandas as pd

#Cargamos datos desde Kaggle
def kaggle_dataset():
    kaggle.api.authenticate()

    kaggle.api.dataset_download_file(
        dataset = 'andradaolteanu/country-mapping-iso-continent-region',
        file_name = 'continents2.csv',
        path = './Practica 6/datasets',  #Estoy ejecutando el script desde el directorio de todas las practicas
        force=True)

#Al igual que con la practica pasada, muchos paises tienen nombres distintos en este dataset
#entonces tuve que cambiarles el nombre para que concuerden y se pudiera hacer el Join
def country_dict():
    #Varios paises en este nuevo dataset tienen nombres distintos a los que vienen el viejo dataset
    #entonces el diccionario esta para cambiar los nombres de estos paises en este dataset
    #para que coincidan con el dataset original
    mapping = {
        "Virgin Islands (British)" : "British Virgin Islands",
        "Brunei Darussalam" : "Brunei",
        "Côte D'Ivoire" : "Cote d'Ivoire",
        "Falkland Islands (Malvinas)" : "Falkland Islands",
        "Guinea Bissau" : "Guinea-Bissau",
        "Réunion" : "Reunion",
        "Saint Barthélemy" : "Saint Barthelemy",
        'Saint Helena, Ascension and Tristan da Cunha' : 'Saint Helena',
        "Virgin Islands (U.S.)" : "United States Virgin Islands",
    }

    return mapping
    
def main():
    kaggle_dataset()

    direccion_actual = os.path.dirname(__file__)
    df_original = pd.read_csv(os.path.join(direccion_actual, '../../Practica 5/datasets/births-and-deaths_joined.csv'))
    df_new = pd.read_csv(os.path.join(direccion_actual, '../../Practica 6/datasets/continents2.csv'))

    #Renombro la columna con los nombres de los paises para que no quede una columna duplicada despues del Join
    df_new = df_new.rename(columns={"name" : "Entity"})

    #Agarro solo estas columnas del dataset
    df_new = df_new[["Entity", "region", "sub-region", "intermediate-region"]]

    #Realizo el mapeo de nombres de paises
    df_new["Entity"] = df_new["Entity"].replace(country_dict())

    join_df = pd.merge(df_original, df_new, left_on=["Entity"], right_on=["Entity"], how="left")

    join_df.to_csv(os.path.join(direccion_actual, '../datasets/births-and-deaths_continents.csv'), index=False)

if __name__ == "__main__":
    main()