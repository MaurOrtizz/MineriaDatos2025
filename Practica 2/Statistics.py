import os
import pandas as pd

#Funcion para generar los nombres de las columnas para que esten todas las estadisticas descriptivas
# para nacimientos y muertes
def col_names(grouping):
    agg_functions = ["Sum", "Count", "Mean", "Min", "Max", "Var", "Std", "Mode", "Kurt"]
    
    cols = []
    cols.extend(grouping)

    for name in ["Births", "Deaths"]:
        for func in agg_functions:
            cols.append(f"{name}_{func}")
    
    return cols

#Funcion para sacar la primer moda o si no hay, imprimir "No Mode"
#Esto lo hice porque me estaba marcando error en la agrupacion porque series.mode() devuelve todo los datos
# si no hay moda y no me deja hacer la agrupacion con datos no escalares
def get_mode(series):
    modes = series.mode()
    if len(series.mode()) == len(series) or len(modes) == 0:
        return "No Mode"
    else:
        return modes.iloc[0]

def analysis_grouping(df, grouping):
    #Regularizar los datos, porque unas las mande como string y otras como lista de strings
    if isinstance(grouping, str):
        grouping = [grouping]

    df_bygroup = df.groupby(grouping).agg({
        "Births_Combined": ["sum", "count", "mean", "min", "max", "var", "std", get_mode, pd.Series.kurt],
        "Deaths_Combined": ["sum", "count", "mean", "min", "max", "var", "std", get_mode, pd.Series.kurt]
    })
    
    df_bygroup = df_bygroup.reset_index()

    df_bygroup.columns = col_names(grouping)
    
    print("ESTADISTICAS DESCRIPTIVAS DEL DATASET AGRUPADO POR: ", grouping)
    print(df_bygroup.head())

    to_csv(df_bygroup, grouping if isinstance(grouping, str) else '_'.join(grouping))
    
def to_csv(df_bygroup, grouping):    
    df_bygroup.to_csv(f'Practica 2/births-and-deaths-statistics_({grouping}).csv', index=False)

def main():
    direccion_actual = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(direccion_actual, '../Practica 1/births-and-deaths_cleaned.csv'))

    #Realizamos las agrupaciones por entidad (paises y regiones), entidad y tipo de dato (estimado o proyeccion) y por año
    analysis_grouping(df, "Entity")
    analysis_grouping(df, ["Entity", "Data_Type"])
    analysis_grouping(df, "Year")

    # Estadisticas descriptivas del dataset sin agrupacion
    df_stats = df.agg({
    "Births_Combined": ["sum", "count", "mean", "min", "max", "var", "std", get_mode, pd.Series.kurt],
    "Deaths_Combined": ["sum", "count", "mean", "min", "max", "var", "std", get_mode, pd.Series.kurt]
    })
    print(df_stats)

if __name__ == "__main__":
    main()
