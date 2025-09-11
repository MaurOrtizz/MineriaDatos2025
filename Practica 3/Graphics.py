import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import textwrap

#Grafica 1: Linea de tiempo de la suma de nacimientos y muertes por año
def plot_by_year(dir, filename):
    df = pd.read_csv(os.path.join(dir, filename))

    plt.figure(figsize=(10, 6))

    #Graficamos la suma de nacimientos y muertes por año
    plt.plot(df['Year'], df['Births_Sum'], label='Nacimientos Combinados', color='blue')
    plt.plot(df['Year'], df['Deaths_Sum'], label='Muertes Combinadas', color='red')

    plt.xlabel('Año')
    plt.ylabel('Cantidad')

    plt.title('Suma de Nacimientos y Muertes estimadas y proyectadas desde 1950 a 2100')

    #Formato del plot: Agregue una linea para indicar cuando pasan a ser proyecciones, agregue la leyenda y una cuadricula
    plt.axvline(x=2023.5, color='black', linestyle='dotted', alpha=0.7, label='Cambio de estimados a proyecciones')
    plt.legend()
    plt.grid(True)

    #Formato del eje y, para que aparezca como numeros separados por comas y 
    #un margen para que se vea bien el label del eje y
    formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(left=0.15)

    plt.savefig(os.path.join(dir, '../Practica 3/img/births-and-deaths-over-years.png'))
    #plt.show()

#Grafica 2: Linea de tiempo de de suma nacimientos y muertes por región geográfica y por año
def plot_by_region(dir, filename):
    df = pd.read_csv(os.path.join(dir, filename))

    #Para esta grafica solo tomo en cuenta las regions que son continentes
    #que son las que contienen (UN) al final del nombre
    df = df[(df['Entity_Type'] == 'Region') & (df['Entity'].str.contains('UN', na=False))]

    #Tomamos la lista de las regiones para hacer un ciclo for
    regions = df['Entity'].unique()

    plt.plot(figsize=(14, 8))
    #Nos vamos de region en region graficando los nacimientos y muertes de cada una
    for region in regions:
        region_df = df[df['Entity'] == region]

        plt.plot(region_df['Year'], region_df['Births_Combined'], label=f'Nacimientos - {region}')
        plt.plot(region_df['Year'], region_df['Deaths_Combined'], label=f'Muertes - {region}', linestyle='dashed')

    plt.xlabel('Año')
    plt.ylabel('Cantidad')
    plt.title('Suma de Nacimientos y Muertes por Región desde 1950 a 2100')
    
    #Posiciono la leyenda abajo del grafico porque si no se termina cortando
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    plt.grid(True)

    #Formato del eje y, para que aparezca como numeros separados por comas y 
    #un margen para que se vea toda la leyenda
    plt.subplots_adjust(bottom=0.25)
    formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(os.path.join(dir, '../Practica 3/img/births-and-deaths-by-regions.png'), bbox_inches='tight', dpi=300)
    #plt.show()

#Grafica 3: Linea de tiempo de cambio porcentual en muertes por region geográfica y por año
#Esta basada en la grafica 2
def plot_by_region_yoy(dir, filename):
    df = pd.read_csv(os.path.join(dir, filename))

    # Solo usamos regiones que son continentes (terminan en UN)
    df = df[(df['Entity_Type'] == 'Region') & (df['Entity'].str.contains('UN', na=False))]

    regions = df['Entity'].unique()

    plt.figure(figsize=(14, 8))

    for region in regions:
        region_df = df[df['Entity'] == region].sort_values(by='Year')

        # Calculamos el cambio interanual
        region_df['Deaths_YoY'] = region_df['Deaths_Combined'].pct_change() * 100
        
        plt.plot(region_df['Year'], region_df['Deaths_YoY'], label=f'Δ Muertes - {region}')

    plt.xlabel('Año')
    plt.ylabel('Cambio Porcentual comparado al año anterior (%)')
    plt.title('Cambio Porcentual de Muertes por Región de 1950 a 2100')
    plt.axvline(x=2023.5, color='black', linestyle='dotted', alpha=0.7, label='Cambio de estimados a proyecciones')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    plt.grid(True)
    
    plt.subplots_adjust(bottom=0.3)
    formatter = FuncFormatter(lambda x, _: f"{int(x):,}%")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.axvline(x=2023.5, color='black', linestyle='dotted', alpha=0.7, label='Cambio de estimados a proyecciones')

    plt.savefig(os.path.join(dir, '../Practica 3/img/births-and-deaths-by-regions-yoy.png'), bbox_inches='tight', dpi=300)
    #plt.show()

#Grafica 4: Boxplot de distribucion de nacimientos y muertes en paises por decadas 
def dist_boxplot(dir, filename):
    df = pd.read_csv(os.path.join(dir, filename))

    #Creamos una columna de decadas y eliminamos a los registros de regiones
    df['Decade'] = (df['Year'] // 10) * 10
    df.drop(df[df['Entity_Type'] != 'Country'].index, inplace=True)

    #Creamos un boxplot para nacimientos y otro para muertes
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True, gridspec_kw={'wspace': 0.05})
    
    #Boxplot de Nacimientos
    df.boxplot(column="Births_Combined", by="Decade", ax=ax1, grid=False, patch_artist=True)
    ax1.set_title("Nacimientos por Decada")
    ax1.set_xlabel("Decadas")
    ax1.set_ylabel("Cantidad")
    
    #Boxplot de Muertes
    df.boxplot(column="Deaths_Combined", by="Decade", ax=ax2, grid=False, patch_artist=True)
    ax2.set_title("Muertes por Decada")
    ax2.set_xlabel("Decadas")
    ax2.set_ylabel("Cantidad")
    
    #Usamos una escala logaritmica debido a que hay una diferencia muy grande entre
    #los outliers mas altos y la caja, es mas que nada para que sea mas facil de ver y entender
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    
    #Cambios el formato default de las cantidades de nacimientos/muertes
    #para que aparezcan como numeros separados por comas 
    formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)

    #Agregamos un titulo para el grafico de dos boxplots
    plt.suptitle("Distribución de Nacimientos y Muertes por Década")
    
    #Guardamos la imagen
    save_path = os.path.join(dir, "../Practica 3/img/births-and-deaths-distribution-boxplot.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    #plt.show()

def income_barchart(dir, filename):
    df = pd.read_csv(os.path.join(dir, filename))

    #Ocupamos solo las regiones que estan basadas en nivel de dearrollo
    df = df[((df['Entity'].str.contains('countries', na=False)) | (df['Entity'].str.contains('regions', na=False)))]

    #Creamos las posiciones para las barras
    x = range(len(df))
    width = 0.35

    #Graficamos dos barras para cada region
    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], df['Births_Sum'], width=width, label='Nacimientos', alpha=0.8)
    plt.bar([i + width/2 for i in x], df['Deaths_Sum'], width=width, label='Muertes', alpha=0.8)

    #Formato de numeros separados por comas
    formatter = FuncFormatter(lambda x, _: f'{int(x):,}')
    plt.gca().yaxis.set_major_formatter(formatter)

    #Labels y titulos, mostramos leyenda y habilitamos que los labels del eje x esten en mas de un renglon
    #para aumentar la legibilidad
    plt.xticks(ticks=x, labels=df['Entity'], rotation=45)
    plt.xlabel('Regiones por Nivel de Desarrollo')
    plt.ylabel('Cantidad')
    plt.title('Diferencias en sumas de nacimientos y muertes segun el nivel de desarrollo')
    plt.legend()
    plt.tight_layout()
    wrapped_labels = [textwrap.fill(label, width=15) for label in df['Entity']]
    plt.xticks(ticks=x, labels=wrapped_labels, rotation=30, ha='right')

    plt.savefig(os.path.join(dir, '../Practica 3/img/births-and-deaths-by-income.png'), bbox_inches='tight', dpi=300)
    #plt.show()

def boxplot_datatype_yoy(dir, filename, births=True):
    df = pd.read_csv(os.path.join(dir, filename))

    #Quitamos la region de America porque no tiene estimados de nacimientos
    df = df[df['Entity'] != 'Americas (UN)']

    #Calculamos el cambio porcentual comparado con el año anterior
    df['Births_YoY'] = df.groupby('Entity')['Births_Combined'].pct_change() * 100
    df['Deaths_YoY'] = df.groupby('Entity')['Deaths_Combined'].pct_change() * 100

    #Eliminamos los valores NaN, en esto caso es el del primer año
    df_clean = df.dropna(subset=['Births_YoY', 'Deaths_YoY'])

    #Se divide el dataset en dos y se crean los labels dependiendo de los datos que se vieren ver
    if births:
        estimates = df_clean[df_clean['Data_Type'] == 'Estimate']['Births_YoY']
        projections = df_clean[df_clean['Data_Type'] == 'Projection']['Births_YoY']
        ylabel = 'Cambio porcentual en nacimientos comparado con el año anterior'
        title = 'Diferencias en distribución en los cambios porcentuales por año de los estimados y proyecciones de nacimientos'
        savename = 'births-distribution-boxplot-yoy.png'
    else:
        estimates = df_clean[df_clean['Data_Type'] == 'Estimate']['Deaths_YoY']
        projections = df_clean[df_clean['Data_Type'] == 'Projection']['Deaths_YoY']
        ylabel = 'Cambio porcentual en muertes comparado con el año anterior'
        title = 'Diferencias en distribución en los cambios porcentuales por año de los estimados y proyecciones de muertes'
        savename = 'deaths-distribution-boxplot-yoy.png'

    #Creamos el boxplot comparando estimados y proyecciones
    _, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([estimates, projections], labels=['Estimados', 'Proyecciones'], patch_artist=True)

    #Le ponemos al boxplot el formato segun si es de nacimientos o muertes
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y')
    formatter = FuncFormatter(lambda x, _: f"{int(x):,}%")
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(os.path.join(dir, f'../Practica 3/img/{savename}'), bbox_inches='tight', dpi=300)
    #plt.show()

def main():
    direccion_actual = os.path.dirname(__file__)

    plot_by_year(direccion_actual, '../Practica 2/births-and-deaths-statistics_(Year).csv')

    plot_by_region(direccion_actual, '../Practica 1/births-and-deaths_cleaned.csv')

    plot_by_region_yoy(direccion_actual, '../Practica 1/births-and-deaths_cleaned.csv')

    dist_boxplot(direccion_actual, '../Practica 1/births-and-deaths_cleaned.csv')

    income_barchart(direccion_actual, '../Practica 2/births-and-deaths-statistics_(Entity).csv')

    boxplot_datatype_yoy(direccion_actual, '../Practica 1/births-and-deaths_cleaned.csv', births=True)

if __name__ == "__main__":
    main()