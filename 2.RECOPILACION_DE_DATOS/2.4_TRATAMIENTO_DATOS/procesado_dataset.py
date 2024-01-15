import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#Ruta a modificar segun el equipo donde corra el modelo
directorio_principal = '/Users/jesusvillamuelas/Documents/'

# Construir la ruta
csv_PREVISIONES = os.path.join(directorio_principal, "PREVISIONES_PROCESADOS.csv")
csv_SIMEL = os.path.join(directorio_principal, "ENERGIA_PROCESADOS.csv")
csv_DATASET = os.path.join(directorio_principal, "DATASET.csv")

# Cargar los datos desde los archivos CSV que comprondan el dataset
df1 = pd.read_csv(csv_PREVISIONES,sep=';')
df2 = pd.read_csv(csv_SIMEL, sep=';')

#Selecciono solo campos a utilizar en el DataSet final

df1_1 = df1[['FESESION','HORA','ENERGIA']]
print(df1_1)
df2_2 = df2[['Fecha','Period','Energy_SIMEL']]
print(df2_2)
#Homogeneizo las columnas entre ambos archivos para sean coincidentes para merge
df1_1.rename(columns={"FESESION":"Fecha","HORA":"Period",'ENERGIA':'PREVISION'}, inplace=True)
df2_2.rename(columns={"Energy_SIMEL":"E_SIMEL"}, inplace=True)
print(df1_1)
print(df2_2)
# Unir los dos conjuntos de datos en función de la fecha
df= pd.merge(df1_1, df2_2,on=["Fecha","Period"], how="inner")

#Compruebo si existen huecos que puedan dar probelmas en calculos siguientes
#No debe al hace un inner join
df.isnull().sum()

#Dropeo datasets ya procesados
del df1, df2, df1_1, df2_2

#Calculo del desvio entre dos variables principales
df["DESVIO"] = df["PREVISION"]- df["E_SIMEL"]

#Calculo variables dummy 

#flag si la prevision en superior a la produccion registrada
df['f_PREV_HIGH'] = df['PREVISION'] > df['E_SIMEL']
#print(df[df['f_PREV_HIGH'] == True])  

#flag si la prevision en inferior a la produccion registrada
df['f_PREV_LOW'] = df['PREVISION'] < df['E_SIMEL']
#print(df[df['f_PREV_LOW'] == True])  

#flag si la instalacion está produciendo o parada
df['f_RUN'] = df['E_SIMEL'] > 0

#flag instalacion arrancada y prevision a 0
df['f_RUN_NOPREV'] = np.where((df['E_SIMEL'] > 0) & (df['PREVISION'] == 0), True, False)

#flag instalacion prevision pero no arranca
df['f_RUN_NO_ARRANCA'] = np.where((df['PREVISION'] > 0) & (df['E_SIMEL'] <= 0), True, False)
print(df)


#Volcado a csv
df.to_csv(csv_DATASET)




