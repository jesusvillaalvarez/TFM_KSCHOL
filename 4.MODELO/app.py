from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ingreso-previsiones', methods=['GET', 'POST'])
def ingreso_previsiones():
    if request.method == 'POST':
        fecha = request.form.get('fecha')
        # Obtener los datos del formulario para cada hora
        previsiones = [request.form.get(f'prevision{hora}', '0') for hora in range(1, 25)]

        # Crear un DataFrame con los datos de previsión
        df_previsiones = pd.DataFrame({
            'Fecha': [fecha] * 24,
            'Period': list(range(1, 25)),
            'PREVISION': previsiones,
            'E_SIMEL': [0] * 24,
            'DESVIO': [pd.NA] * 24,  # Usar pd.NA para valores nulos
            'f_PREV_HIGH': [pd.NA] * 24,
            'f_RUN': [pd.NA] * 24,
            'Dia_Semana': datetime.strptime(fecha, '%Y-%m-%d').weekday() + 1,
            'Es_fin_semana': datetime.strptime(fecha, '%Y-%m-%d').weekday() >= 5,
            'Año': datetime.strptime(fecha, '%Y-%m-%d').year,
            'Mes': datetime.strptime(fecha, '%Y-%m-%d').month,
            'Día': datetime.strptime(fecha, '%Y-%m-%d').day
        })

        # Usar la fecha de las previsiones para generar el nombre del archivo
        nombre_archivo_unico = f"DATOS_PARA_PREVISIONES_{fecha.replace('-', '')}.csv"
        ruta_archivo_nuevo = f'C:/Users/Windows 10/Desktop/MASTER DATASCIENCE/TFM/PROYECTO TFM/{nombre_archivo_unico}'

        # Guardar el DataFrame de previsiones al nuevo archivo CSV
        df_previsiones.to_csv(ruta_archivo_nuevo, index=False)

        return redirect(url_for('index'))

    return render_template('ingreso_previsiones.html')

if __name__ == '__main__':
    app.run(debug=True)










