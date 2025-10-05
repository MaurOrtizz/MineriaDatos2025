import kaggle
import pandas as pd
import os


def kaggle_dataset():
    kaggle.api.authenticate()

    #Descargo los 4 csv que son parte de este dataset, pero solo me interesan 2 de estos
    kaggle.api.dataset_download_files(
        dataset = 'programmerrdai/global-demographic-dynamics-population-trends',
        path = './Practica 5/datasets',  #Estoy ejecutando el script desde el directorio de todas las practicas
        unzip=True)

def delete_files(direccion_actual):
    #Borro los 2 csv que no me interesan de este dataset
    for file in ['US_PopAgeStruct_20230713030811.csv', 'US_PopGR_20230713030828.csv']:
        direccion = os.path.join(direccion_actual, f"datasets/{file}")

        if(os.path.exists(direccion)):
            os.remove(direccion)

#Limpieza del csv que empieza con "US_PopTotal"
def cleaning_total(direccion_actual):
    df = pd.read_csv(os.path.join(direccion_actual, "datasets/US_PopTotal_20230713030810.csv"))

    #Solo tomo estas columnas del dataset
    df = df[["Year", "Economy Label", "Absolute value in thousands", "Urban population as percentage of total population"]]

    nulls = df["Absolute value in thousands"].isnull().sum()
    print("Suma de nulls: ", nulls)

    #Si sale que hay nulls, pero hay un patron en los paises que hay nulos
    #Lo explico mas a detalle en DatasetJoin.py
    if nulls > 0:
        nully = df[df["Absolute value in thousands"].isnull()]
        
        null_countries = nully["Economy Label"].unique()
        print(null_countries)

        print(df.isnull().sum())
    
    return df

#Limpieza del csv que empieza con "US_PopDependency"
def cleaning_dependency(direccion_actual):
    df = pd.read_csv(os.path.join(direccion_actual, "datasets/US_PopDependency_20230713030812.csv"))

    #Solo tomo estas columnas del dataset
    df = df[~df["Series Label"].isin(['Old-age dependency ratio', 'Child dependency ratio'])]

    df = df[["Year", "Economy Label", "Persons per hundred persons aged 15-64"]]

    nulls = df["Persons per hundred persons aged 15-64"].isnull().sum()
    print("Suma de nulls: ", nulls)

    #Si sale que hay nulls, pero hay un patron en los paises que hay nulos
    #Lo explico mas a detalle en DatasetJoin.py
    if nulls > 0:
        nully = df[df["Persons per hundred persons aged 15-64"].isnull()]
        
        null_countries = nully["Economy Label"].unique()
        print(null_countries)

        print(df.isnull().sum())
    
    return df

def country_dict():
    #Varios paises en este nuevo dataset tienen nombres distintos a los que vienen el viejo dataset
    #entonces el diccionario esta para cambiar los nombres de estos paises en este dataset
    #para que coincidan con el dataset original
    mapping = {
        "Bolivia (Plurinational State of)" : "Bolivia",
        '"Bonaire, Sint Eustatius and Saba"' : "Bonaire Sint Eustatius and Saba",
        "Brunei Darussalam" : "Brunei",
        "China, Hong Kong SAR" : "Hong Kong",
        "China, Macao SAR" : "Macao",
        "China, Taiwan Province of" : "Taiwan",
        '"Congo, Dem. Rep. of the"' : "Democratic Republic of Congo",
        "Côte d'Ivoire" : "Cote d'Ivoire",
        "Curaçao" : "Curacao",
        "Falkland Islands (Malvinas)" : "Falkland Islands",
        '"France, metropolitan"' : "France",
        "Réunion" : "Reunion",
        "Iran (Islamic Republic of)" : "Iran",
        '"Korea, Dem. People\'s Rep. of"' : "North Korea",
        '"Korea, Republic of"' : "South Korea",
        "Lao People's Dem. Rep." : "Laos",
        "Micronesia (Federated States of)" : "Micronesia (country)",
        "Netherlands (Kingdom of the)" : "Netherlands",
        "Saint Barthélemy" : "Saint Barthelemy",
        "Syrian Arab Republic" : "Syria",
        '"Tanzania, United Republic of"' : "Tanzania",
        "Timor-Leste" : "East Timor",
        "Türkiye" : "Turkey",
        "United States of America excluding Puerto Rico and United States Virgin Islands" : "United States",
        "Viet Nam" : "Vietnam",
        "Wallis and Futuna Islands" : "Wallis and Futuna",
        "Africa" : "Africa (UN)",
        "Northern America" : "Northern America (UN)",
        "Latin America and the Caribbean" : "Latin America and the Caribbean (UN)",
        "Asia" : "Asia (UN)",
        "Americas" : "Americas (UN)",
        "Europe" : "Europe"
    }

    return mapping

def drop_duplicates(df):
    #El nuevo dataset tiene datos para toda Francia (Francia continetal + 5 regiones de ultramar)
    #y Francia metropolitana (continental). El dataset viejo solo toma en cuenta Francia metropolitana
    #porque hay datos para cada una de las regiones de ultramar, entonces elimino las filas que tengan el nombre
    #'Francia' para que haya consistencia en los datos
    df.drop(df[df['Economy Label'] == "France"].index, inplace=True)

    #Tambien ocurre lo mismo con Estados Unidos, en este caso con Puerto Rico y las islas virgenes
    df.drop(df[df['Economy Label'] == "United States of America"].index, inplace=True)

def column_names():
    #Esto es simplemente darle valores mas descriptivos a las columnas del dataset
    dict = {"Absolute value in thousands" : "Population in thousands",
    "Urban population as percentage of total population" : "Urban Population (%)",
    "Persons per hundred persons aged 15-64" : "Dependency Ratio (%)",
    "Economy Label" : "Entity"}
    
    return dict    

def main():
    direccion_actual = os.path.dirname(__file__)
    
    kaggle_dataset()
    delete_files(direccion_actual)

    df_total = cleaning_total(direccion_actual)

    df_dependency = cleaning_dependency(direccion_actual)

    #Junto los dos csv
    df = pd.merge(df_total, df_dependency, left_on=["Economy Label", "Year"], right_on=["Economy Label", "Year"], how="inner")

    mapping = country_dict()

    drop_duplicates(df)
    
    df['Economy Label'] = df['Economy Label'].replace(mapping)

    new_columns = column_names()
    
    df.rename(columns=new_columns, inplace=True)

    df.to_csv("Practica 5/datasets/PopulationInfo_cleaned.csv", index=False)

if __name__ == "__main__":
    main()