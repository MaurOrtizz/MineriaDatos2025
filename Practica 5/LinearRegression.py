import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import os

def regression(df, x, y):
    #Creamos una variable que tenga la columna del valor de x
    #y otra para guardar la constante del intercepto
    x_constant = sm.add_constant(df[x])

    #Creacion del modelo de regresion lineal
    model = sm.OLS(df[y], x_constant).fit()

    print("=====REGRESION LINEAL=====")
    print(f"Variable dependiente (y): {y}")
    print(f"Variable independiente (x): {x}")
    
    #Impresion de datos del modelo de regresion lineal
    print("\n")
    print(model.summary())
    print("\n")

    #Graficamos la relacion entre la x y la y
    plot(model, df, x, y)

def plot(model, df, x, y):
    #Ordenamos los valores en el dataframe en relacion a x
    df = df.sort_values(by=x)
    x_constant = sm.add_constant(df[x])
    
    #Agregamos los puntos en la grafica
    plt.scatter(df[x], df[y], color="blue")
    #Agregamos la linea de regresion a la grafica
    plt.plot(df[x], model.predict(x_constant), color="red")

    #Agregamos titulos a los ejes
    plt.xlabel(x)
    plt.ylabel(y)

    #Titulo de la grafica
    plt.title(f"Regresión Lineal - {y} explicada por {x}")
    #Mostramos cuadricula
    plt.grid(True)
    #Para que las imagenes guardadas no corten el texto de los titulos
    plt.tight_layout()
    #Guardamos imagen y mostramos
    plt.savefig(os.path.join(os.path.dirname(__file__), f'../Practica 5/img/{x}_{y}_regression.png'), bbox_inches="tight")
    plt.show()

def main():
    direccion_actual = os.path.dirname(__file__)
    #Para este practica combine mi dataset original con uno que tiene informacion sobre la poblacion de cada pais,
    #además de su porcentaje de población urbana y tasa de dependencia
    #Carga y limpieza del nuevo dataset -> NewDatasetLoadingCleaning.py
    #Combinacion del dataset nuevo y viejo -> DatasetJoin.py
    #Ambos estan en la carpeta que se llama "new_dataset_scripts"
    df = pd.read_csv(os.path.join(direccion_actual, '../Practica 5/datasets/births-and-deaths_joined.csv'))

    #Creo las columnas de tasas de nacimientos y tasa de muertes
    df["Birth_Rate"] = df["Births_Combined"] / (df["Population in thousands"])
    df["Death_Rate"] = df["Deaths_Combined"] / (df["Population in thousands"])

    #Aqui solo tomo los datos mundiales para que las graficas sean mas legibles y
    #debido a que tuve problemas con las regresiones usando los datos de cada pais y
    #me aparecia una R2 mas baja de lo que deberia
    df = df[df['Entity'] == 'World']
    
    #Hipotesis:
        #Vamos a ver que entre menos poblacion urbana, va a haber una tasa de dependencia mas alta
        #por la cantidad de menos de edad, debido a que paises subdesarrollados tienen un % de poblacion
        #urbana mas bajo y una tasa de fertilidad mas alta, PERO
        #entre suba la tasa de poblacion urbana a niveles muy altos vamos a ver un incremento
        #en la tasa de dependencia 
    #Interpretacion: 
        #Existe una correlacion muy alta entre las dos columnas, el incremento en la tasa de dependencia en
        #niveles de poblacion urbana muy altos no fue tan drastico como pensaba, pero esto puede ser porque 
        #son datos globales, y en promedio el mundo aun no esta tan urbanizado
        #Algo interesante que ocurre en la grafica es que vemos valores mas altos de dependencia cuando
        #la poblacion urbana esta alrededor del 32% y 42%, esto puede ser por el periodo de desarrollo economico
        #explosivo y baby boom que hubo globalmente despues de la SGM, a mi parecer.
    regression(df, x="Urban Population (%)", y="Dependency Ratio (%)")

    #Hipotesis:
        #Entre el porcentaje de personas en areas urbanas sea mas altos, la tasa de nacimientos va a ser mas baja
        #Este es una tendencia global muy conocida, entonces me imagino que va a tener una R2 muy alta
        #Muchos expertos tienen sus propias teorias de porque ocurre esto, pero la que mas se me hace correcta
        #es que en zonas rurales el tener muchos hijos es una ventaja porque pueden ayudar en las labores del dia a dia
        #Y en una ciudad, tener hijos no resulta en el mismo beneficio y es a un costo mas elevado porque las ciudades son
        #mas caras que las areas rurales (generalmente)
    #Interpretacion:
        #Confirme mi hipotesis, los datos siguen la linea de regresion casi perfectamente, a excepcion de un pedazo al principio
        #con niveles bajos de poblacion urbana pero que aumentan y una tasa de nacimientos que es mas alta de lo que deberia
        #Puede que esto signifique que el año u otra variable tambien tenga efecto ligero en la tasa de nacimientos
    regression(df, x="Urban Population (%)", y="Birth_Rate")

    #Hipotesis:
        #La tasa de nacimiento es muy alta cuando hay una tasa de dependencia muy alta, pero entre que va bajando
        #la tasa de dependencia a los valores altos pero que no son los extremos, empezamos a ver mas variacion en los valores
        #de la tasa de nacimiento, porque en un pais (o año) en el que hay una tasa de dependencia alta esto puede ser
        #por un incremento en la cantidad de nacimientos o por el envejecimiento de la poblacion
        #La correlacion es mas debil en esta, puede que sea un valor bajo de R2 y que se pueda explicar mejor
        #usando otro tipo de regresion
    #Interpretacion:
        #Hubo menos variacion de la que esperaba pero tambien de otro tipo que no esperaba
        #Si vemos que entre que suba la tasa de dependencia, vemos valores similares con tasas de nacimientos muy distintas
        #Pero tambien vemos esto con las tasas de dependencia bajas, realmente la parte que es mas estable son los valores
        #en el medio. En teoria esto no deberia pasar, porque el unico que hay en las tasas de dependencias bajas
        #es que es un pais (o año) en el que las personas que crecieron de un baby boom tienen menos hijos que sus padres
        #Y aun asi, vemos que en años con tasas de dependencia menores a 55 (que al parecer fueron varios) las tasas de nacimientos
        #van de 15 a 20 por mil personas, y en valores con tasas de 55 a 60 la tasa de nacimientos se mantiene casi estable
        #en valores cercanos a 15 por mil personas 
    regression(df, x="Dependency Ratio (%)", y="Birth_Rate")

    #Hipotesis:
        #Entre mas sea la tasa de dependencia mas alto la tasa de muertes, porque los paises con una poblacion mas joven
        #tienden a ser mas pobres y con menor calidad de salud, entonces deberia de ocurrir lo mismo con los años. Tambien
        #en el caso de que sea tasa alta de dependencia por la cantidad de personas de la tercera edad, se esperaría una
        #tasa alta de muertes
    #Interpretacion: 
        #Sorprendentemente, esta fue la regresion con el valor de R2 mas bajo. Vemos que en los años con tasas de dependencia muy altos
        #hay tanto valores bajos como altos de muertes. Esto tal vez puede ser porque mientras envejece la poblacion, la tasa de dependencia
        #no baja, solo se transfiere a otro tipo de dependencia (menores de edad -> mayores de edad), por lo que diría que los valores 
        #altos y bajos son de las proyecciones a futuro y los estimados de los años mas antiguos en el dataset
    regression(df, x="Dependency Ratio (%)", y="Death_Rate")

if __name__ == '__main__':
    main()