import os
import pandas as pd
import numpy as np

#Funcion de renombrar columnas
#Las columnas numericas tienen nombres muy largos y con espacios asi que esta funcion las renombra
def renombrar_columnas(columna):
    dict = {
        'Births - Sex: all - Age: all - Variant: estimates': 'Births_Estimates',
        'Births - Sex: all - Age: all - Variant: medium': 'Births_Projections',
        'Deaths - Sex: all - Age: all - Variant: estimates': 'Deaths_Estimates',
        'Deaths - Sex: all - Age: all - Variant: medium': 'Deaths_Projections'
    }
    return dict.get(columna, columna)

#Funcion para estandarizar valores faltantes a NaN
#Por si acaso hay valores faltantes que no sean NaN, la funcion los convierte a NaN
def missing_a_nan(df):
    df.replace(['', ], np.nan, inplace=True)

#Funcion de combinacion de columnas
#Esta funcion combina las columnas que son de nacimientos y las que son de muertes para solo tener una de cada tipo
#Para mantener la distincion entre si es una proyecion o un estimado basado en datos reales
#Agrego una columna que especifica que tipo de dato es
def combinacion_columnas(df):
    df['Births_Combined'] = np.where(df['Births_Estimates'].notna(), df['Births_Estimates'], df['Births_Projections'])
    df['Deaths_Combined'] = np.where(df['Deaths_Estimates'].notna(), df['Deaths_Estimates'], df['Deaths_Projections'])

    df['Data_Type'] = np.where(df['Births_Estimates'].notna(), 'Estimate', 'Projection')

    df.drop(columns=['Births_Estimates', 'Births_Projections', 'Deaths_Estimates', 'Deaths_Projections'], inplace=True)

#Funcion de cambiar columnas a entero
#Las columnas numericas son de tipo flotante y algunas tienen decimales
#Que las columnas sean flotantes no tiene sentido en el contexto de nacimientos y muertes
#Por lo que hay que redondear las columnas y pasarlas a entero
def pasar_a_entero(df, columnas):
    for col in columnas:
        df[col] = df[col].round().astype('Int64')

#Funcion para agregar tipo de entidad
#El dataset incluye datos de todos los miembros de la ONU mas aparte varias regiones agrupadas por continente o nivel de desarrollo
#Estas regiones luego me podrian servir en el analisis, asi que no las quito pero no tienen valor en la columna 'Code'
#La columna 'Code' tampoco la quiero quitar de momento porque luego voy a tener que complementar este dataset con otro
#Para poder distinguir entre paises y regiones (fuera de checar los valores NaN en 'Code') creo una columna que especifica
#si el registro es de un pais o una region
def tipo_entidad(df):
    df['Entity_Type'] = np.where(df['Code'].notna(), 'Country', 'Region')

#Funcion para eliminar los registros de entidades
#Por el momento solo la uso con el Vaticano, razonamiento viene en Main 
def droppear_entidades(df, nombre_entidad):
    df.drop(df[df['Entity'] == nombre_entidad].index, inplace=True)

def main():
    direccion_actual = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(direccion_actual, 'births-and-deaths-projected-to-2100.csv'))

    df = df.rename(columns=renombrar_columnas)

    combinacion_columnas(df)

    missing_a_nan(df)
    
    pasar_a_entero(df, columnas=['Births_Combined', 'Deaths_Combined'])

    tipo_entidad(df)

    #Note que el Vaticano en ciertos años tiene una cantidad muy grande de nacimientos registrados, 
    #especialmente en las proyecciones. Nadie nace en el Vaticano porque no hay hospitales ahi, pero algunos hijos de
    #oficiales son registrados como nacidos en el Vaticano si nacieron en Roma, pero esto no es 
    #lo mismo a que nazcan en el Vaticano.
    #Tambien si llegara a agregar una columna de poblacion cuando complemente el dataset,
    #los datos del Vaticano serian muy diferentes a los del resto del mundo por su poblacion de <1000 personas. 
    #Por eso prefiero eliminar las filas completamente.
    droppear_entidades(df, nombre_entidad='Vaticano')

    df.to_csv(os.path.join(direccion_actual, 'births-and-deaths_cleaned.csv'), index=False)

if __name__ == '__main__':
    main()