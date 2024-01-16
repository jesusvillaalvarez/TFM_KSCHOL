import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
file_path = '/Users/jesusvillamuelas/Documents/DATASET.csv'
dataset = pd.read_csv(file_path)

# Seleccionar la variable objetivo y las características
y = dataset['f_RUN_NO_ARRANCA']
X = dataset[['PREVISION']]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarización de las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Evaluar el modelo
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



           precision    recall  f1-score   support

       False       0.97      1.00      0.98      4845
        True       0.00      0.00      0.00       169

    accuracy                           0.97      5014
   macro avg       0.48      0.50      0.49      5014
weighted avg       0.93      0.97      0.95      5014
