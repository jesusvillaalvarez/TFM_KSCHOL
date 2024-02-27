#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Tratamiento de datos
import pandas as pd
import numpy as np

# Gráficos
import matplotlib.pyplot as plt
from matplotlib import style

# Preprocesado y modelado
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders
from sklearn.preprocessing import OrdinalEncoder
import missingno
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


# In[2]:


import pandas as pd

# Load the CSV file into a DataFrame
df_central = pd.read_csv("C:/Users/Windows 10/Desktop/MASTER DATASCIENCE/TFM/df_central_2_1.csv")

# Display the first few rows of the dataframe to verify its content
df_central.head()


# In[3]:


# Eliminar las columnas innecesarias
df_central = df_central.drop(columns=['Unnamed: 0', 'DESVIO', 'f_PREV_HIGH', 'f_PREV_LOW'])


# In[4]:


# Convertir la columna 'Fecha' a datetime para facilitar el filtrado
df_central['Fecha'] = pd.to_datetime(df_central['Fecha'])

# Dividir el DataFrame en dos según las fechas especificadas
df_inicio = df_central[df_central['Fecha'] <= '2023-10-31']
df_final = df_central[df_central['Fecha'] >= '2023-11-05']

# Eliminar la columna 'Fecha' de ambos DataFrames
df_inicio = df_inicio.drop(columns=['Fecha'])
df_final = df_final.drop(columns=['Fecha'])


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Preparar las características (X) y la variable objetivo (y)
X = df_inicio.drop('E_SIMEL', axis=1)
y = df_inicio['E_SIMEL']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características (esto es comúnmente necesario para modelos de deep learning)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificar las dimensiones de los conjuntos de datos para asegurarnos de que todo está correcto
(X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape)


# In[6]:


import tensorflow as tf
from tensorflow.keras import layers, models

# Definir la arquitectura del modelo
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Capa de salida para predicción de un valor continuo
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2, batch_size=32, verbose=1)

# Evaluar el modelo con el conjunto de prueba
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Loss en el conjunto de prueba: {test_loss}, MAE: {test_mae}")


# In[7]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# In[8]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor

# Configurar el imputador MICE con GradientBoostingRegressor como el estimador
mice_imputer = IterativeImputer(estimator=GradientBoostingRegressor(
                                    n_estimators=100,
                                    max_depth=10,
                                    min_samples_split=4,
                                    min_samples_leaf=2,
                                    max_features='sqrt'),
                                    max_iter=10, random_state=42)

# Preparar los datos para el entrenamiento del imputador MICE (usamos df_inicio sin E_SIMEL)
X_mice = df_inicio.drop(columns=['E_SIMEL'])

# Entrenar el imputador MICE
mice_imputer.fit(X_mice)

# Confirmar que el imputador ha sido entrenado
"Imputador MICE entrenado con éxito."


# In[9]:


# Seleccionamos las dilas del dia 5 del df_final para hacer el tratamiento de la variable f_RUN

df_final_05_11 = df_final[(df_final['Año'] == 2023) & (df_final['Mes'] == 11) & (df_final['Día'] == 5)]

df_final_05_11_para_imputar = df_final_05_11.drop(['E_SIMEL'], axis=1)

df_final_05_11_para_imputar[['f_RUN']] = np.nan  # Primero convertimos la columna f_RUN a NaN

# En la variable creada para la imputación, predecimos los valores para las columnas siguientes

valores_imputados = mice_imputer.transform(df_final_05_11_para_imputar[['Period', 'PREVISION', 'f_RUN', 'Dia_Semana', 'Es_fin_semana', 'Año', 'Mes', 'Día']])

# Realmente solo queremos imputar la columna f_RUN, por lo tanto cogemos solo los valores imputados a la columna en concreto

valores_imputados_f_RUN = np.where(valores_imputados[:, 2]> 0.2, 1, 0)

df_final_05_11.loc[:, 'f_RUN'] = valores_imputados_f_RUN


# In[10]:


# Preparar los datos de df_final_05_11 para la predicción
X_final_05_11 = df_final_05_11.drop(['E_SIMEL'], axis=1)

# Normalizar los datos de X_final_05_11 utilizando el mismo scaler que para los datos de entrenamiento
X_final_05_11_scaled = scaler.transform(X_final_05_11)

# Realizar las predicciones de E_SIMEL con el modelo de deep learning
e_simel_predicciones = model.predict(X_final_05_11_scaled)

e_simel_predicciones = np.maximum(e_simel_predicciones, 0)

# Mostrar las primeras 5 predicciones
e_simel_predicciones[:25]


# In[11]:


# Convertir el array de predicciones a una lista para facilitar la asignación
predicciones_lista = e_simel_predicciones.flatten().tolist()

# Asignar las predicciones a df_final_05_11
df_final_05_11['Prediccion_E_SIMEL'] = predicciones_lista

# Mostrar las primeras filas para verificar
df_final_05_11[['E_SIMEL', 'PREVISION', 'Prediccion_E_SIMEL']].head(25)



# In[12]:


suma_real_05 = df_final_05_11['E_SIMEL'].sum()
suma_predicha_05 = df_final_05_11['Prediccion_E_SIMEL'].sum()
suma_prevision_05 = df_final_05_11['PREVISION'].sum()


if suma_real_05 != 0:
    desviacion_porcentual = 100 * (suma_predicha_05 - suma_real_05) / suma_real_05
else:
    desviacion_porcentual = float('inf')  # en caso de división por cero, retorna un valor especial para que no nos dé error



if suma_real_05 != 0:
    desviacion_porcentual_prevision = 100 * (suma_prevision_05 - suma_real_05) / suma_real_05
else:
    desviacion_porcentual_prevision = float('inf')  # en caso de división por cero, retorna un valor especial para que no nos dé error

print("Suma real: ", suma_real_05)
print("Suma predicha: ", suma_predicha_05)
print("Desviación porcentual: ", desviacion_porcentual, "%")
print("Suma previsión: ", suma_prevision_05)
print("Desviación porcentual: ", desviacion_porcentual_prevision, "%")


# In[13]:


# Una vez tenemos la primera predicción, actualizamos df_inicio con los datos del día 5. Así estamos simulando como sería
# el proceso de predicción en tiempo real

datos_dia_5 = df_final[(df_final['Año'] == 2023) & (df_final['Mes'] == 11) & (df_final['Día'] == 5)]
df_inicio_actualizado = pd.concat([df_inicio, datos_dia_5])

# Preparar los datos para el modelo de deep learning
X_actualizado = df_inicio_actualizado.drop('E_SIMEL', axis=1)
y_actualizado = df_inicio_actualizado['E_SIMEL']


# Normalizar las características
X_total_scaled = scaler.fit_transform(X_actualizado)  # Utilizamos fit_transform para reajustar el scaler a los nuevos datos

# Reentrenar el modelo de deep learning con todos los datos
model.fit(X_total_scaled, y_actualizado, verbose = 1)

mice_imputer.fit(df_inicio_actualizado[['Period', 'PREVISION', 'f_RUN', 'Dia_Semana', 'Es_fin_semana', 'Año', 'Mes', 'Día']])


# In[14]:


import numpy as np

# Seleccionamos las dilas del dia 5 del df_final para hacer el tratamiento de la variable f_RUN

df_final_06_11 = df_final[(df_final['Año'] == 2023) & (df_final['Mes'] == 11) & (df_final['Día'] == 6)]

df_final_06_11_para_imputar = df_final_06_11.drop(['E_SIMEL'], axis=1)

df_final_06_11_para_imputar[['f_RUN']] = np.nan  # Primero convertimos la columna f_RUN a NaN

# En la variable creada para la imputación, predecimos los valores para las columnas siguientes

valores_imputados = mice_imputer.transform(df_final_06_11_para_imputar[['Period', 'PREVISION', 'f_RUN', 'Dia_Semana', 'Es_fin_semana', 'Año', 'Mes', 'Día']])

# Realmente solo queremos imputar la columna f_RUN, por lo tanto cogemos solo los valores imputados a la columna en concreto

valores_imputados_f_RUN = np.where(valores_imputados[:, 2]> 0.2, 1, 0)

df_final_06_11.loc[:, 'f_RUN'] = valores_imputados_f_RUN


# Verificamos que los valores han sido imputados correctamente
# df_final_05_11.head(25)

# Preparar los datos de df_final_05_11 para la predicción
X_final_06_11 = df_final_06_11.drop(['E_SIMEL'], axis=1)

# Normalizar los datos de X_final_05_11 utilizando el mismo scaler que para los datos de entrenamiento
X_final_06_11_scaled = scaler.transform(X_final_06_11)

# Realizar las predicciones de E_SIMEL con el modelo de deep learning
e_simel_predicciones_06 = model.predict(X_final_06_11_scaled)

e_simel_predicciones_06 = np.maximum(e_simel_predicciones_06, 0)

# Mostrar las primeras 5 predicciones
# e_simel_predicciones[:25]

# Convertir el array de predicciones a una lista para facilitar la asignación
predicciones_lista = e_simel_predicciones_06.flatten().tolist()

# Asignar las predicciones a df_final_05_11
df_final_06_11['Prediccion_E_SIMEL'] = predicciones_lista

# Mostrar las primeras filas para verificar
df_final_06_11[['E_SIMEL', 'PREVISION', 'Prediccion_E_SIMEL']].head(25)


# In[15]:


suma_real_06 = df_final_06_11['E_SIMEL'].sum()
suma_predicha_06 = df_final_06_11['Prediccion_E_SIMEL'].sum()
suma_prevision_06 = df_final_06_11['PREVISION'].sum()


if suma_real_06 != 0:
    desviacion_porcentual = 100 * (suma_predicha_06 - suma_real_06) / suma_real_06
else:
    desviacion_porcentual = float('inf')  # en caso de división por cero, retorna un valor especial para que no nos dé error



if suma_real_06 != 0:
    desviacion_porcentual_prevision = 100 * (suma_prevision_06 - suma_real_06) / suma_real_06
else:
    desviacion_porcentual_prevision = float('inf')  # en caso de división por cero, retorna un valor especial para que no nos dé error

print("Suma real: ", suma_real_06)
print("Suma predicha: ", suma_predicha_06)
print("Desviación porcentual: ", desviacion_porcentual, "%")
print("Suma previsión: ", suma_prevision_06)
print("Desviación porcentual: ", desviacion_porcentual_prevision, "%")


# In[16]:


# Creamos un función para agilizar el proceso de actualización, reentreno de los modelos, imputación, predicción y cálculo de las métricas

def predecir_y_actualizar_para_un_dia(dia_actual, dia_siguiente,  mes, año, df_inicio_actualizado, df_final, modelo_deep, imputador):
    """
    Función para actualizar el conjunto de entrenamiento con los datos reales de un día específico,
    realizar la imputación para el día siguiente y predecir los valores de E_SIMEL para ese día.

    Args:
    dia_actual (int): Día actual para el que se actualizarán los datos.
    dia_siguiente (int): Datos del día que queremos hacer las imputaciones y la predicción
    mes (int): Mes del día actual.
    ano (int): Año del día actual.
    df_inicio_actualizado (DataFrame): DataFrame actualizado con los datos hasta el día anterior.
    df_final (DataFrame): DataFrame con los datos a predecir.
    modelo_rf (RandomForestRegressor): Modelo de Random Forest entrenado.
    imputador (IterativeImputer): Imputador MICE entrenado.

    Returns:
    DataFrame: DataFrame con las predicciones para el día siguiente.
    DataFrame: DataFrame actualizado con los datos reales del día actual.
    """
    # Actualización de df_actualizado con los datos de dia_actual

    datos_dia_actual = df_final[(df_final['Año'] == año) & (df_final['Mes'] == mes) & (df_final['Día'] == dia_actual)]
    df_inicio_actualizado = pd.concat([df_inicio_actualizado, datos_dia_actual])

    # Reentrenamos los modelos

    # Una vez tenemos la primera predicción, actualizamos df_inicio con los datos del día 5. Así estamos simulando como sería
    # el proceso de predicción en tiempo real



    # Preparar los datos para el modelo de deep learning
    X_actualizado = df_inicio_actualizado.drop('E_SIMEL', axis=1)
    y_actualizado = df_inicio_actualizado['E_SIMEL']

    
    
    # Normalizar las características
    X_actualizado_scaled = scaler.transform(X_actualizado, y_actualizado)

    # Reentrenar el modelo de deep learning con todos los datos
    modelo_deep.fit(X_actualizado_scaled, y_actualizado, verbose = 1)

    imputador.fit(df_inicio_actualizado[['Period', 'PREVISION', 'f_RUN', 'Dia_Semana', 'Es_fin_semana', 'Año', 'Mes', 'Día']])

    

    

      # Imputación de valores a la columna f_RUN

    df_dia_siguiente = df_final[(df_final['Año'] == año) & (df_final['Mes'] == mes) & (df_final['Día'] == dia_siguiente)]
    df_dia_siguiente_para_imputar = df_dia_siguiente.drop(['E_SIMEL'], axis=1)
    df_dia_siguiente_para_imputar[['f_RUN']] = np.nan  
    
    valores_imputados = imputador.transform(df_dia_siguiente_para_imputar)
    
    df_dia_siguiente.loc[:,'f_RUN'] = np.where(valores_imputados[:, 2] > 0.2, 1, 0) 

    
    
    # Preparar los datos de df_final_05_11 para la predicción
    X_prediccion = df_dia_siguiente.drop(['E_SIMEL'], axis=1)

    # Normalizar los datos de X_final_05_11 utilizando el mismo scaler que para los datos de entrenamiento
    X_prediccion_scaled = scaler.transform(X_prediccion)

    # Realizar las predicciones de E_SIMEL con el modelo de deep learning
    predicted_e_simel = model.predict(X_prediccion_scaled)

    predicted_e_simel = np.maximum(predicted_e_simel, 0)

    # Mostrar las primeras 5 predicciones
    # e_simel_predicciones[:25]

    # Convertir el array de predicciones a una lista para facilitar la asignación
    predicciones_lista = predicted_e_simel.flatten().tolist()

    df_predicciones = df_dia_siguiente[['Año', 'Mes', 'Día', 'PREVISION', 'E_SIMEL']].copy()

    # Asignar las predicciones a df_final_05_11
    df_predicciones['Prediccion_E_SIMEL'] = predicciones_lista

    # Mostrar las primeras filas para verificar
    df_predicciones[['E_SIMEL', 'PREVISION', 'Prediccion_E_SIMEL']].head(25)


    

    # Cálculo de las métricas

    mse = mean_squared_error(df_predicciones['E_SIMEL'], df_predicciones['Prediccion_E_SIMEL'])
    r2 = r2_score(df_predicciones['E_SIMEL'], df_predicciones['Prediccion_E_SIMEL'])
    mae = mean_absolute_error(df_predicciones['E_SIMEL'], df_predicciones['Prediccion_E_SIMEL'])


    return df_predicciones, df_inicio_actualizado, mse, r2, mae


# In[17]:


# Llamamos a la función
dia_actual = 6    # Actualización de datos con los que reentrenamos los modelos
dia_siguiente = 7    # Preparación de datos para la imputación y predicción
df_predicciones_07_11, df_inicio_actualizado, mse_07_11, r2_07_11, mae_07_11 = predecir_y_actualizar_para_un_dia(dia_actual, dia_siguiente, 11, 2023, df_inicio_actualizado, df_final, model, mice_imputer)
print("MSE:", mse_07_11, "R²:", r2_07_11, "MAE:", mae_07_11)


# In[18]:


# Y como en la predicción anterior calculamos los sumatorios y los porcentajes de desviación

suma_real_07_11 = df_predicciones_07_11['E_SIMEL'].sum()
suma_predicha_07_11 = df_predicciones_07_11['Prediccion_E_SIMEL'].sum()
suma_prevision_07_11 = df_predicciones_07_11['PREVISION'].sum()


if suma_real_07_11 != 0:
    desviacion_porcentual = 100 * (suma_predicha_07_11 - suma_real_07_11) / suma_real_07_11
else:
    desviacion_porcentual = float('inf')  



if suma_real_07_11 != 0:
    desviacion_porcentual_prevision = 100 * (suma_prevision_07_11 - suma_real_07_11) / suma_real_07_11
else:
    desviacion_porcentual_prevision = float('inf')  
    

print("Suma real: ", suma_real_07_11)
print("Suma predicha: ", suma_predicha_07_11)
print("Desviación porcentual: ", desviacion_porcentual, "%")
print("Suma previsión: ", suma_prevision_07_11)
print("Desviación porcentual: ", desviacion_porcentual_prevision, "%")


# In[19]:


# Llamamos a la función. Actualización de datos reales del día 7 para reentrenar los modelos
# Preparación de los datos del día posterior, en este caso del día 8, para la imputación y predicción

dia_actual = 7
dia_siguiente = 8

df_predicciones_08_11, df_inicio_actualizado, mse_08_11, r2_08_11, mae_08_11 = predecir_y_actualizar_para_un_dia(dia_actual, dia_siguiente, 11, 2023, df_inicio_actualizado, df_final, model, mice_imputer)
print("MSE:", mse_08_11, "R²:", r2_08_11, "MAE:", mae_08_11)


# In[20]:


# Sumatorios y porcentajes

suma_real_08_11 = df_predicciones_08_11['E_SIMEL'].sum()
suma_predicha_08_11 = df_predicciones_08_11['Prediccion_E_SIMEL'].sum()
suma_prevision_08_11 = df_predicciones_08_11['PREVISION'].sum()


if suma_real_08_11 != 0:
    desviacion_porcentual = 100 * (suma_predicha_08_11 - suma_real_08_11) / suma_real_08_11
else:
    desviacion_porcentual = float('inf')  



if suma_real_08_11 != 0:
    desviacion_porcentual_prevision = 100 * (suma_prevision_08_11 - suma_real_08_11) / suma_real_08_11
else:
    desviacion_porcentual_prevision = float('inf')  
    

print("Suma real: ", suma_real_08_11)
print("Suma predicha: ", suma_predicha_08_11)
print("Desviación porcentual: ", desviacion_porcentual, "%")
print("Suma previsión: ", suma_prevision_08_11)
print("Desviación porcentual: ", desviacion_porcentual_prevision, "%")


# In[21]:


# Seguimos con el mismo proceso anterior. Llamamos a la función con los días específicos que queremos actualizar, imputar y predecir.

dia_actual = 8
dia_siguiente = 9

df_predicciones_09_11, df_inicio_actualizado, mse_09_11, r2_09_11, mae_09_11 = predecir_y_actualizar_para_un_dia(dia_actual, dia_siguiente, 11, 2023, df_inicio_actualizado, df_final, model, mice_imputer)
print("MSE:", mse_09_11, "R²:", r2_09_11, "MAE:", mae_09_11)


# In[22]:


# Sumatorios y procentajes

suma_real_09_11 = df_predicciones_09_11['E_SIMEL'].sum()
suma_predicha_09_11 = df_predicciones_09_11['Prediccion_E_SIMEL'].sum()
suma_prevision_09_11 = df_predicciones_09_11['PREVISION'].sum()


if suma_real_09_11 != 0:
    desviacion_porcentual = 100 * (suma_predicha_09_11 - suma_real_09_11) / suma_real_09_11
else:
    desviacion_porcentual = float('inf')  



if suma_real_09_11 != 0:
    desviacion_porcentual_prevision = 100 * (suma_prevision_09_11 - suma_real_09_11) / suma_real_09_11
else:
    desviacion_porcentual_prevision = float('inf')  
    

print("Suma real: ", suma_real_09_11)
print("Suma predicha: ", suma_predicha_09_11)
print("Desviación porcentual: ", desviacion_porcentual, "%")
print("Suma previsión: ", suma_prevision_09_11)
print("Desviación porcentual: ", desviacion_porcentual_prevision, "%")


# In[23]:


# Llamamos a la función
dia_actual = 9
dia_siguiente = 10

df_predicciones_10_11, df_inicio_actualizado, mse_10_11, r2_10_11, mae_10_11 = predecir_y_actualizar_para_un_dia(dia_actual,dia_siguiente, 11, 2023, df_inicio_actualizado, df_final, model, mice_imputer)
print("MSE:", mse_10_11, "R²:", r2_10_11, "MAE:", mae_10_11)


# In[24]:


# Sumatorios y porcentajes

suma_real_10_11 = df_predicciones_10_11['E_SIMEL'].sum()
suma_predicha_10_11 = df_predicciones_10_11['Prediccion_E_SIMEL'].sum()
suma_prevision_10_11 = df_predicciones_10_11['PREVISION'].sum()


if suma_real_10_11 != 0:
    desviacion_porcentual = 100 * (suma_predicha_10_11 - suma_real_10_11) / suma_real_10_11
else:
    desviacion_porcentual = float('inf')  



if suma_real_10_11 != 0:
    desviacion_porcentual_prevision = 100 * (suma_prevision_10_11 - suma_real_10_11) / suma_real_10_11
else:
    desviacion_porcentual_prevision = float('inf')  

print("Suma real: ", suma_real_10_11)
print("Suma predicha: ", suma_predicha_10_11)
print("Desviación porcentual: ", desviacion_porcentual, "%")
print("Suma previsión: ", suma_prevision_10_11)
print("Desviación porcentual: ", desviacion_porcentual_prevision, "%")


# In[25]:


# Llamamos a la función

dia_actual = 10
dia_siguiente = 13

df_predicciones_13_11, df_inicio_actualizado, mse_13_11, r2_13_11, mae_13_11 = predecir_y_actualizar_para_un_dia(dia_actual,dia_siguiente, 11, 2023, df_inicio_actualizado, df_final, model, mice_imputer)
print("MSE:", mse_13_11, "R²:", r2_13_11, "MAE:", mae_13_11)


# In[26]:


# Sumas y porcentajes

suma_real_13_11 = df_predicciones_13_11['E_SIMEL'].sum()
suma_predicha_13_11 = df_predicciones_13_11['Prediccion_E_SIMEL'].sum()
suma_prevision_13_11 = df_predicciones_13_11['PREVISION'].sum()

if suma_real_13_11 != 0:
    desviacion_porcentual = 100 * (suma_predicha_13_11 - suma_real_13_11) / suma_real_13_11
else:
    desviacion_porcentual = float('inf')  



if suma_real_13_11 != 0:
    desviacion_porcentual_prevision = 100 * (suma_prevision_13_11 - suma_real_13_11) / suma_real_13_11
else:
    desviacion_porcentual_prevision = float('inf')  

print("Suma real: ", suma_real_13_11)
print("Suma predicha: ", suma_predicha_13_11)
print("Desviación porcentual: ", desviacion_porcentual, "%")
print("Suma previsión: ", suma_prevision_13_11)
print("Desviación porcentual: ", desviacion_porcentual_prevision, "%")


# In[27]:


# Llamamos a la función
dia_actual = 13
dia_siguiente = 14

df_predicciones_14_11, df_inicio_actualizado, mse_14_11, r2_14_11, mae_14_11 = predecir_y_actualizar_para_un_dia(dia_actual,dia_siguiente, 11, 2023, df_inicio_actualizado, df_final, model, mice_imputer)
print("MSE:", mse_14_11, "R²:", r2_14_11, "MAE:", mae_14_11)


# In[28]:


# Sumatorios y porcentajes

suma_real_14_11 = df_predicciones_14_11['E_SIMEL'].sum()
suma_predicha_14_11 = df_predicciones_14_11['Prediccion_E_SIMEL'].sum()
suma_prevision_14_11 = df_predicciones_14_11['PREVISION'].sum()

if suma_real_14_11 != 0:
    desviacion_porcentual = 100 * (suma_predicha_14_11 - suma_real_14_11) / suma_real_14_11
else:
    desviacion_porcentual = float('inf')  



if suma_real_14_11 != 0:
    desviacion_porcentual_prevision = 100 * (suma_prevision_14_11 - suma_real_14_11) / suma_real_14_11
else:
    desviacion_porcentual_prevision = float('inf')  

print("Suma real: ", suma_real_14_11)
print("Suma predicha: ", suma_predicha_14_11)
print("Desviación porcentual: ", desviacion_porcentual, "%")
print("Suma previsión: ", suma_prevision_14_11)
print("Desviación porcentual: ", desviacion_porcentual_prevision, "%")


# In[29]:


# Llamamos a la función

dia_actual = 14
dia_siguiente = 15

df_predicciones_15_11, df_inicio_actualizado, mse_15_11, r2_15_11, mae_15_11 = predecir_y_actualizar_para_un_dia(dia_actual,dia_siguiente, 11, 2023, df_inicio_actualizado, df_final, model, mice_imputer)
print("MSE:", mse_15_11, "R²:", r2_15_11, "MAE:", mae_15_11)


# In[30]:


# Sumas y porcentajes

suma_real_15_11 = df_predicciones_15_11['E_SIMEL'].sum()
suma_predicha_15_11 = df_predicciones_15_11['Prediccion_E_SIMEL'].sum()
suma_prevision_15_11 = df_predicciones_15_11['PREVISION'].sum()

if suma_real_15_11 != 0:
    desviacion_porcentual = 100 * (suma_predicha_15_11 - suma_real_15_11) / suma_real_15_11
else:
    desviacion_porcentual = float('inf')  



if suma_real_15_11 != 0:
    desviacion_porcentual_prevision = 100 * (suma_prevision_15_11 - suma_real_15_11) / suma_real_15_11
else:
    desviacion_porcentual_prevision = float('inf')  

print("Suma real: ", suma_real_15_11)
print("Suma predicha: ", suma_predicha_15_11)
print("Desviación porcentual: ", desviacion_porcentual, "%")
print("Suma previsión: ", suma_prevision_15_11)
print("Desviación porcentual: ", desviacion_porcentual_prevision, "%")


# In[31]:


# Concatenamos todos los Dataframes que contienen las prediccions, previsiones y datos reales para calcular las méstricas en conjunto

df_predicciones_totales = pd.concat([df_final_05_11, df_final_06_11, df_predicciones_07_11, 
                                     df_predicciones_08_11, df_predicciones_09_11,df_predicciones_10_11, 
                                     df_predicciones_13_11, df_predicciones_14_11, df_predicciones_15_11])


# In[32]:


# Cálculo de las métricas para ver si nos indican si mejoran los errores entren la predicción (predicted_E_SIMEL), previsión (PREVISION)y producción real(E_SIMEL)

def calcular_metricas(df):
    mae = mean_absolute_error(df['E_SIMEL'], df['Prediccion_E_SIMEL'])
    mse = mean_squared_error(df['E_SIMEL'], df['Prediccion_E_SIMEL'])
    r2 = r2_score(df['E_SIMEL'], df['Prediccion_E_SIMEL'])
    return mae, mse, r2

# Métricas para la predicciones

mae_pred, mse_pred, r2_pred = calcular_metricas(df_predicciones_totales)

# Cambiamos la columna de predicción por la de previsión

df_previsiones = df_predicciones_totales.copy()
df_previsiones['Prediccion_E_SIMEL'] = df_previsiones['PREVISION']

# Métricas para la previsión

mae_prev, mse_prev, r2_prev = calcular_metricas(df_previsiones)

# Visualizamos los resultados

print("MAE Predicciones: ", mae_pred)
print("MSE Predicciones: ", mse_pred)
print("R² Predicciones: ", r2_pred)
print("MAE Previsiones: ", mae_prev)
print("MSE Previsiones: ", mse_prev)
print("R² Previsiones: ", r2_prev)


# In[33]:


# Para saber si llevamos una acumulado positivo de predicciones vs previsiones sobre la producción real,
# calculamos las diferencias absolutas

# Diferencia absoluta entre la predición y la producción real

df_predicciones_totales['diff_pred_real'] = abs(df_predicciones_totales['Prediccion_E_SIMEL'] - df_predicciones_totales['E_SIMEL'])

# Diferencia absoluta entre la previsión y la producción real

df_predicciones_totales['diff_prev_real'] = abs(df_predicciones_totales['PREVISION'] - df_predicciones_totales['E_SIMEL'])


# sumamos todos los valores de las columnas que queremo comparar

suma_e_simel = df_predicciones_totales['E_SIMEL'].sum()
sumas_totales_predicciones = df_predicciones_totales['Prediccion_E_SIMEL'].sum()
sumas_previsiones = df_predicciones_totales['PREVISION'].sum()


# Calculamos las diferencias entre la prediccion y la previsión respecto la producción real E_SIMEL

diferencia_prediccion_vs_produccion_real = abs(sumas_totales_predicciones - suma_e_simel)
diferencia_prevision_vs_produccion_real = abs(sumas_previsiones - suma_e_simel)


# Imprimimos los resultados para poder visualizar si mejoramos las previsiones a lo largo de todas las predicciones.

print(f"Suma de los valores en la columna E_SIMEL: {suma_e_simel}")
print(f"Suma de las predicciones: {sumas_totales_predicciones}")
print(f"Suma de las previsiones : {sumas_previsiones}")


print(f"Diferencia entre predicciones totales y E_SIMEL total: {diferencia_prediccion_vs_produccion_real}")
print(f"Diferencia entre previsiones y E_SIMEL total: {diferencia_prevision_vs_produccion_real}")

diferencia = diferencia_prediccion_vs_produccion_real - diferencia_prevision_vs_produccion_real

if diferencia_prediccion_vs_produccion_real > diferencia_prevision_vs_produccion_real:
    print(f"No mejoramos la predicción respecto la PREVISION real en: {diferencia}, por lo tanto, con este modelo, no estamos mejorando las previsiones.")
else:
    print(f"La predicción es MEJOR que la previsión en: {-diferencia} unidades, por lo tanto, cumplimos nuestro objetivo de mejorar la PREVISIÓN.")


# In[ ]:


suma = df_final['PREVISION'].sum()


# In[ ]:


suma

