import requests
import pandas as pd
import json

# Reemplaza 'TU_CLAVE_DE_API' con tu clave real de la API de AEMET
api_key = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqZXN1cy52aWxsYS5hbHZhcmV6QGdtYWlsLmNvbSIsImp0aSI6ImVkM2JkN2M0LWU4NDctNDQwZi04ODc2LTlhNmI2YTM1NWY1MCIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzAxMjU4MTU3LCJ1c2VySWQiOiJlZDNiZDdjNC1lODQ3LTQ0MGYtODg3Ni05YTZiNmEzNTVmNTAiLCJyb2xlIjoiIn0.52akELBtqCXQERxeeoORUzprT8dMbMEHOaKLtjiiiSk'

# Construye la URL de la API para obtener la previsión del tiempo
url = f'https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/horaria/10109?api_key={api_key}'

# Realiza la solicitud a la API
response = requests.get(url)
content = response.json()

# Obtiene los datos y metadatos
r_meta = requests.get(content['metadatos'], verify=False)
metadata = r_meta.json()

r_data = requests.get(content['datos'], verify=False)
data = r_data.json()

# Normaliza el JSON
df_data = pd.json_normalize(data)

# Crea un DataFrame para las predicciones diarias
df_prediccion_dia = pd.json_normalize(df_data["prediccion.dia"])

# Concatenar las columnas del DataFrame
df_prediccion_dia_unificado = pd.concat([df_prediccion_dia[col] for col in df_prediccion_dia.columns], axis=1)

# Seleccionar solo la primera columna del DataFrame final
primera_columna_json = df_prediccion_dia_unificado.iloc[:, 1]

# Convertir la columna de JSON a DataFrame
primera_columna_df = pd.json_normalize(json.loads(primera_columna_json.to_json(orient='records')))

# Crear un nuevo dataset con la primera columna
nuevo_dataset = primera_columna_df

# Supongamos que 'nuevo_dataset' es tu dataset con la primera columna en formato JSON
# Extraer la columna "estadoCielo" del nuevo_dataset
columna_estado_cielo_json = nuevo_dataset['temperatura']

# Convertir la columna de JSON a DataFrame
columna_estado_cielo_df = pd.json_normalize(json.loads(columna_estado_cielo_json.to_json(orient='records')))

# Crear un nuevo dataset con la columna "estadoCielo"
nuevo_dataset_estado_cielo = columna_estado_cielo_df

# Muestra el nuevo dataset con la columna "estadoCielo"
print(nuevo_dataset_estado_cielo)

# Crear un nuevo dataset normalizado con la columna "estadoCielo"
nuevo_dataset_estado_cielo_normalizado = pd.json_normalize(columna_estado_cielo_df.to_dict(orient='records'))

nuevo_dataset_estado_cielo_normalizado = nuevo_dataset_estado_cielo_normalizado.rename(
    columns=lambda x: x.split('.')[-1]
)


# Muestra el nuevo dataset normalizado con la columna "estadoCielo"
print(nuevo_dataset_estado_cielo_normalizado)

# Obtener el número de columnas en el DataFrame
num_columnas = nuevo_dataset_estado_cielo_normalizado.shape[1]

# Crear un nuevo DataFrame para el resultado final
nuevo_dataset_final = pd.DataFrame()

# Iterar sobre las columnas y reorganizar el contenido
for i in range(0, num_columnas, 2):
    nuevo_dataset_final = pd.concat([nuevo_dataset_final, nuevo_dataset_estado_cielo_normalizado.iloc[:, i:i+2].reset_index(drop=True)], axis=0)

# Restablecer los índices del nuevo dataset final
nuevo_dataset_final.reset_index(drop=True, inplace=True)

# Muestra el nuevo dataset final
print(nuevo_dataset_final)