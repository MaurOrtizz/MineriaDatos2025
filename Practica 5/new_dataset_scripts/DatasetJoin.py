import pandas as pd
import os

direccion_actual = os.path.dirname(__file__)

df_original = pd.read_csv(os.path.join(direccion_actual, '../Practica 1/births-and-deaths_cleaned.csv'))
df_new = pd.read_csv(os.path.join(direccion_actual, '../Practica 5/datasets/PopulationInfo_cleaned.csv'))

join_df = pd.merge(df_original, df_new, left_on=["Entity", "Year"], right_on=["Entity", "Year"], how="left")

#El nuevo dataset tiene datos de 1950 a 2050, por lo que para este dataset conjunto elimino los años de 1950 a 2050
#minimo para la practica de la regresion lineal, tener valores nulos me traeria problemas
join_df = join_df[join_df["Year"] <= 2050]

#Aqui lo que descubri es que hay varios nulos en este dataset porque los paises
#estan organizados de manera distinta a mi dataset original
#Este dataset no pone valores para paises que no existian en cierto año
#Por ejemplo, para la union sovietica hay valores de 1950 a 1991 y despues son puros nulos
#y para los miembros de la union sovietica no hay valores antes de 1991
#pasa lo mismo con paises que perdieron territorio de 1950 hasta la fecha (porque un pais se independizo de ellos), 
#como ocurre con Sudan e Indonesia
#Elimine todos los registros de estos paises porque no puedo dividir los valores de paises que eran uniones a sus
#paises constituyentes, no tengo manera de saber que valor le corresponde a cada uno
countries_with_nan = join_df[join_df["Population in thousands"].isna()]["Entity"].unique()

join_df = join_df[~join_df["Entity"].isin(countries_with_nan)]

join_df.to_csv(os.path.join(direccion_actual, 'datasets/births-and-deaths_joined.csv'), index=False)