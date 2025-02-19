# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:20:12 2024

@author: Leticia
"""
"""
Este código realiza una serie de operaciones para modelar y predecir el número de manchas solares utilizando el modelo XGBoost.
"""
# Importo las librerías que usaré:

import sys
import ctypes
import pandas as pd # para cargar el Excel
import numpy as np # para operaciones con vectores 
import matplotlib.pyplot as plt # para gráficos
import matplotlib.dates as mdates #ofrece funciones para formatear y manejar las fechas en los gráficos
from scipy.stats import zscore, skew, kurtosis 
import seaborn as sns # para gráficos
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from itertools import cycle #Para crear iteradores cíclicos
#import pmdarima as pm #para el modelo ARIMA
from sklearn.metrics import mean_squared_error


#Genero un vector cíclico, para la estacionalidad del ciclo solar.Vector de longitud `length`, donde los valores se repiten cíclicamente cada `n_to_repeat`:

def generate_vector(length, n_to_repeat):  # Length:número de meses totales de nuestro DB. n_to_repeat: cada cuánto se repiten 
    values = list(range(1, n_to_repeat+1))  # Genera una lista del 1 al n_to_repeat +1 (11)
    vector = []  # Inicializa el vector vacío
    cycle_iterator = cycle(values)  # Crea un iterador cíclico de los valores
    
    # Agrega elementos al vector hasta que alcance la longitud deseada, el número de meses totales de nuestro DB
    for _ in range(length):
        vector.append(next(cycle_iterator))
    
    return vector


#Creo los títulos del DataFrame, mostrando la relación temporal (`t-n`, `t`, `t+n`):

def Extended_titles(Cn,n_in,n_out):

    Total_titles=[]
    Tl=n_in+n_out+1
 
    for i in range(Tl-1):
        if i<(n_in-1):
            letter_to_add = " (t-" + str(n_in-i-1) +")"
        elif i==(n_in-1):
            letter_to_add = " (t)"
        else:
            letter_to_add = " (t+" + str(i-n_in+1) +")"

        Cn_aux= [word + letter_to_add for word in Cn]
        Total_titles=Total_titles+Cn_aux

    return(Total_titles)

#Convierto la serie temporal en el formato adecuado para el aprendizaje automático, con columnas que representen los retrasos y las futuras predicciones:

def series_to_supervised(data, n_in, n_out, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

#Utilizo los datos de la función anterior para transformar los datos de la serie temporal

def to_supervided2 (yiel,n_in,n_out):

    values = yiel.values # transforma los datos de la serie temporal en un problema de aprendizaje supervisado
    data = series_to_supervised(values, n_in,n_out)
    Cn=list(yiel.columns.values)
    Cn_total=Extended_titles(Cn,n_in,n_out)
    Data = pd.DataFrame(data, columns=Cn_total)
    return Data

#Cargo los datos:
    
Df = pd.read_json('observed-solar-cycle-indices.json')
#print(Df.head())
#print(Df.info())
# Obtiene el último valor de la columna 'time-tag'
ultimo_valor = Df['time-tag'].iloc[-1]

# Imprime el último valor
print("El último valor de 'time-tag' es:", ultimo_valor)

# Verifica el rango de fechas en el DataFrame después de to_supervided2

#Calculo estadisticas descriptivas
#Media
mean_values = Df[['ssn','smoothed_ssn']].mean()
print("Media:\n", mean_values)

#Desviacion estandar
std_dev = Df[['ssn','smoothed_ssn']].std()
print("Desviación Estándar:\n", std_dev)

#Varianza

variance = Df[['ssn','smoothed_ssn']].var()
print("Varianza:\n", variance)

#Sesgo

skewness = Df[['ssn','smoothed_ssn']].apply(lambda x: skew(x.dropna()))
print("Sesgo (Skewness):\n", skewness)

#Curtosis: Mide la forma de la distribución, especialmente de las colas

kurtosis_vals = Df[['ssn','smoothed_ssn']].apply(lambda x: kurtosis(x.dropna()))
print("Curtosis (Kurtosis):\n", kurtosis_vals)

#Cuartiles

quartiles = Df[['ssn','smoothed_ssn']].quantile([0.25, 0.5, 0.75])
print("Cuartiles:\n", quartiles)

#rango intercuartilico
iqr = quartiles.loc[0.75] - quartiles.loc[0.25]
print("Rango Intercuartílico (IQR):\n", iqr)

#Rango

range_values = Df[['ssn','smoothed_ssn']].apply(lambda x: x.max() - x.min())
print("Rango:\n", range_values)

# Tomar una muestra  del dataset:
#Df_sample = Df.sample(frac=0.01, random_state=1)

# Realizo algunas visualizaciones iniciales: histograma, gráfico de dispersión o la matriz de correlación:
 
#HISTOGRAMA SSN

plt.figure(figsize=(12, 6))
plt.hist(Df['ssn'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Número de Manchas Solares')
plt.ylabel('Frecuencia')
plt.title('Distribución del Número de Manchas Solares')
plt.grid(True)
plt.show()

#GRAFICOS DE DISPERSION

#ssn vs timetag
Df['Date'] = pd.to_datetime(Df['time-tag'])  
Df = Df.sort_values(by='Date')




#Calculo la media móvil y el Z-score para detectar posibles anomalías:
    
#Cálculo de la Media Móvil:
Df['SMA'] = Df['ssn'].rolling(window=12, center=True).mean()  # Media móvil con ventana de 12 meses

#Cálculo del Z-Score:
Df['Z-Score'] = zscore(Df['ssn'])  # Calcula el Z-Score para detectar anomalías

#Visualización:
plt.figure(figsize=(12, 8))

# Gráfico de dispersión de los datos originales
plt.scatter(Df['Date'], Df['ssn'], color='blue', alpha=0.6, label='Número de Manchas Solares')

# Añadir línea de media móvil al gráfico
plt.plot(Df['Date'], Df['SMA'], color='red', label='Media Móvil (12 meses)')

# Añadir color para anomalías basadas en Z-Score
plt.scatter(Df['Date'], Df['ssn'], c=abs(Df['Z-Score']), cmap='coolwarm', label='Z-Score', alpha=0.6, edgecolors='k')

# Formatear etiquetas de fechas
plt.gca().xaxis.set_major_locator(mdates.YearLocator(11))  # Mostrar etiquetas de año cada año
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formato del año

# Rotar las etiquetas para que se vean mejor
plt.xticks(rotation=45)

# Etiquetas y título
plt.xlabel('Fecha')
plt.ylabel('Número de Manchas Solares')
plt.title('Número de Manchas Solares a lo Largo del Tiempo')
plt.colorbar(label='Z-Score')  # Barra de color para el Z-Score
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



#MATRIZ DE CORRELACIÓN

Df_numeric = Df.select_dtypes(include=[np.number])

# Manejar datos faltantes si es necesario
Df_numeric = Df_numeric.fillna(Df_numeric.mean())

# Calcular la matriz de correlación
correlation_matrix = Df_numeric.corr()

# Mostrar la matriz de correlación
print(correlation_matrix)

# Visualizar la matriz de correlación usando un mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.show()

#INTENTO MATRICES CORRELACION DISTINTOS PERIODOS

"""
        1- Ajustamos los formatos de las fechas y nos quedamos con las columnas que queremos
"""# Supongamos que tu DataFrame es `df` y tiene una columna de fechas llamada 'Date'
Df['Date'] = pd.to_datetime(Df['Date'], errors='coerce')  # Asegura que la fecha esté en formato datetime
Df.set_index('Date', inplace=True)


# Asegúrate de que solo las columnas numéricas se utilicen para la correlación
Df_numeric = Df.select_dtypes(include=[np.number])

# Define los periodos para dividir la serie temporal (por ejemplo, cada 11 años)
periodos = pd.date_range(start=Df_numeric.index.min(), end=Df_numeric.index.max(), freq='11Y')

# Crear un diccionario para almacenar las matrices de correlación
correlation_matrices = {}

for i in range(len(periodos)-1):
    start_date = periodos[i]
    end_date = periodos[i+1]
    
    Df_periodo = Df_numeric[start_date:end_date]
    
    # Verifica que el periodo no esté vacío antes de calcular la correlación
    if not Df_periodo.empty:
        # Calcula la matriz de correlación para el periodo
        corr_matrix = Df_periodo.corr()
        correlation_matrices[f"{start_date.year}-{end_date.year}"] = corr_matrix
        
        # Visualizar la matriz de correlación
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f"Matriz de Correlación {start_date.year}-{end_date.year}")
        plt.show()

Df = Df.rename(columns={'ssn': 'Sunspot_Number','smoothed_ssn':'Smoothed_Sunspot_Number'})
Df['Date'] = pd.to_datetime(Df['time-tag']) #nos aseguramos de que este en formato fecha
Df.insert(0, 'Date', Df.pop('Date'))
Df=Df[["Date","Sunspot_Number","Smoothed_Sunspot_Number"]]
Df.replace(-1, np.nan, inplace=True)

Df_A=Df.copy()

"""
        2- Visualizamos los datos
"""

#Transforma los datos de la serie temporal en un formato adecuado para el aprendizaje supervisado utilizando `series_to_supervised` y `Extended_titles`.

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(15, 6))  # Adjust the figure size if needed

# Plot Sunspot_Number against Date with a thinner line
ax.plot(Df['Date'], Df['Sunspot_Number'], color='blue', linestyle='-', linewidth=0.75, label='Sunspot Number')  # Adjust linewidth as needed
ax.plot(Df['Date'], Df['Smoothed_Sunspot_Number'], color='red', linestyle='-', linewidth=1, label='Smoothed Sunspot Number')  # Adjust linewidth as needed

# Set labels and title
ax.set_title('Número de manchas solares a lo largo del tiempo')
ax.set_xlabel('Fecha')
ax.set_ylabel('Número de manchas solares')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show grid
ax.grid(True)

# Add legends
ax.legend()

# Show plot
plt.show()


"""
3- Comenzamos a realizar modificaciones en los datos para preparar el modelo 
"""
# Verifica el rango de fechas en el DataFrame antes de la preparación
print("Rango de fechas en el DataFrame original:")
print(Df['Date'].min(), "a", Df['Date'].max())

n_in=24 # numero de observaciones anteriores utilizadas para predecir
n_out=12 # numero de observaciones anteriores utilizadas a predecir

Df=to_supervided2(Df,n_in,n_out) # ponemos el Df listo para el aprendizaje supervisado
Df = Df.dropna(subset=[f'Date (t-{n_in-1})']) #eliminamos las filas que no nos valen 
Df.set_index('Date (t)', inplace=True)

# Verifica las filas con valores NaN en la columna específica
nan_rows = Df[Df[f'Date (t-{n_in-1})'].isna()]

# Muestra las filas con valores NaN
print("Filas con valores NaN en la columna 'Date...':")
print(nan_rows)



# Revisa el rango de fechas después de la transformación
print("Rango de fechas en el DataFrame supervisado:")
print(Df.index.min(), "a", Df.index.max())

# Asegúrate de que el DataFrame supervisado contiene todas las fechas esperadas
print(Df.head())
print(Df.tail())

n_to_repeat=11*12 # 11 años y 12 meses tiene cada ciclo aproximadamente 
v_sta = generate_vector(len(Df), n_to_repeat) #generamos el vector a repetir

Df.insert(0, 'Index_for_seasonality', v_sta)


# desdoblamos entre target y features

Div_index=Df.columns.get_loc("Smoothed_Sunspot_Number (t)")

features=Df.iloc[:,:Div_index+1].columns.to_list()
features=[element for element in features if "Date" not in element]

targets=Df.iloc[:,Div_index:].columns.to_list()
targets=[element for element in targets if "Sunspot_Number" in element]
targets=[element for element in targets if "Smoothed" not in element]


X=Df[features]
y=Df[targets]

X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')



Cut_Date=pd.Timestamp(1971,1,1)

X_train=X[X.index<Cut_Date]
y_train=y[y.index<Cut_Date]

print("Rango de fechas en el DataFrame completo:")
print(X.index.min(), "a", X.index.max())

X_test=X[X.index>=Cut_Date]
y_test=y[y.index>=Cut_Date]

# Verifica el rango de fechas en los conjuntos de entrenamiento y prueba
print("Rango de fechas en el conjunto de entrenamiento:")
print(X_train.index.min(), "a", X_train.index.max())

print("Rango de fechas en el conjunto de prueba:")
print(X_test.index.min(), "a", X_test.index.max())

#VALIDACIÓN CRUZADA Y AJUSTE DEL MODELO




#Se inicializa el modelo XGBoost
XGB_model = XGBRegressor()

# Define los hiperparámetros a buscar en RandomizedSearchCV
param_grid = {
    'learning_rate': [0.001, 0.01],  # Tasa de aprendizaje
    'max_depth': [3, 5],               # Profundidad máxima del árbol
    'min_child_weight': [1, 3],        # Peso mínimo de un nodo hijo
    'n_estimators': [100, 300],        # Número de árboles (estimadores)
    'lambda': [0.1, 1],                # Regularización L2
    'gamma': [0, 0.1]                  # Reducción mínima de pérdida
}

# Define el esquema de validación cruzada
tscv = TimeSeriesSplit(n_splits=3)

# Configuración de RandomizedSearchCV con la validación cruzada
#random_search = RandomizedSearchCV(
   #estimator=XGB_model,
    #param_distributions=param_grid,       # Usar param_distributions en lugar de param_grid
    #n_iter=50,                            # Número de combinaciones a probar (ajustar según sea necesario)
    #cv=tscv,                              # Validación cruzada con TimeSeriesSplit
    #scoring="neg_mean_squared_error",     # Métrica de evaluación
    #random_state=42,
    #n_jobs=-1, 
    #verbose=2
#)

# Búsqueda de los mejores hiperparámetros
#random_result = random_search.fit(X_train, y_train)

# Obtenemos los mejores hiperparámetros
#best_params = random_result.best_params_
best_params ={'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.01, 'lambda': 0.1, 'gamma': 0}
#print("Mejores Hiperparámetros: ", best_params)

# Entreno el modelo con los mejores parámetros
XGB_model = XGBRegressor(**best_params)
XGB_model.fit(X_train, y_train)

# Realiza las predicciones
y_test_pred = XGB_model.predict(X_test)

# Verifica la forma del array
print(y_test_pred.shape) 


# Convierto y_test_pred en un DataFrame con la fecha y la predicción

# Suponiendo que la primera columna de y_test_pred es la predicción para 'Sunspot Number'
sunspot_number_pred = y_test_pred[:, 0]  # Seleccionamos la primera columna

# Suponiendo que X_test tiene un índice de fechas que quieres usar
prediction_dates = X_test.index + pd.DateOffset(months=12) # Asumiendo que el índice de X_test es la fecha

# Asumiendo que el índice de X_test es la fecha real (t), no (t+12)
prediction_dates = X_test.index  # Las fechas no se desplazan, ya que ya corresponden a t+12 en X_test

# Crear un DataFrame con las fechas y las predicciones
y_test_pred_Df = pd.DataFrame({
    'Date (t+12)': prediction_dates,
    'Sunspot_Number (t+12)_pred': sunspot_number_pred
})


# Suponiendo que `y_test` y `y_test_pred_Df` tienen una columna de fechas como índice o columna

# Asegúrate de que la columna de fecha en `y_test` y `y_test_pred_Df` está bien definida
# Aquí se asume que el índice de `y_test` es la fecha

# Crear la figura y los ejes
plt.figure(figsize=(12, 6))

# Graficar los valores reales usando la fecha
plt.plot(y_test.index, y_test['Sunspot_Number (t+12)'], label='Valor Real', color='blue', marker='o')

# Graficar las predicciones usando la fecha
plt.plot(y_test_pred_Df['Date (t+12)'], y_test_pred_Df['Sunspot_Number (t+12)_pred'], label='Predicción', color='red', marker='o')

# Agregar título y etiquetas a los ejes
plt.title('Comparación de Valores Reales y Predicciones de Manchas Solares')
plt.xlabel('Fecha')
plt.ylabel('Número de Manchas Solares')

# Mostrar leyenda
plt.legend()

# Mostrar la gráfica
plt.grid(True)
plt.show()



print("Longitud de y_test:", len(y_test))
print("Longitud de y_test_pred_Df:", len(y_test_pred_Df))
# Alinear las predicciones con los valores reales
# Esto supone que la primera predicción corresponde a la primera fecha en y_test_pred_Df
y_test_pred_Df.index = y_test.index

# Crear un DataFrame combinando los valores reales y las predicciones
comparison_df = pd.DataFrame({
    'Date (t+12)': y_test.index,
    'Sunspot_Number (t+12)_Real': y_test['Sunspot_Number (t+12)'],
    'Sunspot_Number (t+12)_Pred': y_test_pred_Df['Sunspot_Number (t+12)_pred']
})

# Define la ruta donde quieres guardar el archivo
file_path = 'comparacion_predicciones.xlsx'

# Exporta el DataFrame a un archivo Excel
comparison_df.to_excel(file_path, index=False)

print(f"El archivo ha sido guardado en: {file_path}")
#TÉCNICAS DE EVALUACION DEL RENDIMIENTO DEL MODELO

#Error absoluto medio
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test['Sunspot_Number (t+12)'], y_test_pred_Df['Sunspot_Number (t+12)_pred'])
#print(f'MAE: {mae}')

# Paso 1: Calcular la media de Sunspot_Number (t+12) en el conjunto de entrenamiento
mean_sunspot_number = y_train['Sunspot_Number (t+12)'].mean()

# Paso 2: Crear predicciones del baseline utilizando la media calculada
baseline_predictions = [mean_sunspot_number] * len(y_test)

# Paso 3: Calcular el MAE del baseline
baseline_mae = mean_absolute_error(y_test['Sunspot_Number (t+12)'], baseline_predictions)

# Comparar con el MAE del modelo
print(f'MAE del modelo: {mae}')
print(f'MAE del baseline: {baseline_mae}')

# Opcional: calcular la mejora porcentual del modelo sobre el baseline
improvement = (baseline_mae - mae) / baseline_mae
print(f'Mejoría del modelo sobre el baseline: {improvement:.2%}')


# Compare with your model's MAE
improvement = (baseline_mae - mae) / baseline_mae
print(f'Model improvement over baseline: {improvement:.2%}')

#Error cuadrático medio

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test['Sunspot_Number (t+12)'], y_test_pred_Df['Sunspot_Number (t+12)_pred'])
#print(f'MSE: {mse}')

# Paso 1: Calcular el MSE del modelo
mse_model = mean_squared_error(y_test['Sunspot_Number (t+12)'], y_test_pred_Df['Sunspot_Number (t+12)_pred'])
print(f'MSE del modelo: {mse_model}')

# Paso 2: Calcular la media de la variable objetivo en el conjunto de prueba
mean_sunspot_number = y_test['Sunspot_Number (t+12)'].mean()

# Paso 3: Crear predicciones del baseline utilizando la media calculada
baseline_predictions = [mean_sunspot_number] * len(y_test)

# Paso 4: Calcular el MSE del baseline
mse_baseline = mean_squared_error(y_test['Sunspot_Number (t+12)'], baseline_predictions)
print(f'MSE del baseline: {mse_baseline}')

# Paso 5: Calcular la mejora porcentual del modelo sobre el baseline
improvement = (mse_baseline - mse_model) / mse_baseline
print(f'Mejoría del modelo sobre el baseline: {improvement:.2%}')

#Raiz del error cuadrático medio

rmse = mean_squared_error(y_test['Sunspot_Number (t+12)'], y_test_pred_Df['Sunspot_Number (t+12)_pred'], squared=False)
print(f'RMSE del modelo: {rmse}')

# Calcular el RMSE del baseline
baseline_predictions = [y_test['Sunspot_Number (t+12)'].mean()] * len(y_test)
rmse_baseline = mean_squared_error(y_test['Sunspot_Number (t+12)'], baseline_predictions, squared=False)
print(f'RMSE del baseline: {rmse_baseline}')

range_y_test = y_test['Sunspot_Number (t+12)'].max() - y_test['Sunspot_Number (t+12)'].min()
print(f'Rango de los valores: {range_y_test}')


# Error absoluto medio porcentual (MAPE)
#y_true = y_test['Sunspot_Number (t+12)']
#y_pred = y_test_pred_Df['Sunspot_Number (t+12)_pred']
#mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#print(f'MAPE: {mape}%')

#Tabla estadisticas descriptivas
# Crear un DataFrame con los resultados
stats_df = pd.DataFrame({
    'Estadística': [
        'Media', 
        'Desviación Estándar', 
        'Varianza', 
        'Sesgo (Skewness)', 
        'Curtosis (Kurtosis)', 
        'Cuartil Q1', 
        'Mediana (Q2)', 
        'Cuartil Q3', 
        'Rango Intercuartílico (IQR)', 
        'Rango'
    ],
    'Valor': [
        mean_values,
        std_dev,
        variance,
        skewness,
        kurtosis_vals,
        quartiles.loc[0.25],  # Cuartil Q1
        quartiles.loc[0.5],   # Mediana (Q2)
        quartiles.loc[0.75],  # Cuartil Q3
        iqr,
        range_values
    ]
})

# Convertir el diccionario a un DataFrame
stats_df = pd.DataFrame(stats_df)

# Imprimir la tabla
print(stats_df)

# Convertir el diccionario a un DataFrame
stats_df = pd.DataFrame(stats_df)

# Guardar el DataFrame en un archivo Excel
file_path = 'estadisticas_descriptivas.xlsx'
stats_df.to_excel(file_path, index=False)

# Confirmación
print(f"Tabla guardada en el archivo {file_path}")



# No consigo ajustar bien el ARIMA, hago comparaciones con otros modelos:
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Cargar los datos
Df_comparativa = pd.read_json('observed-solar-cycle-indices.json')
Df_comparativa = Df_comparativa.sort_values('time-tag')

# Definir el tamaño del conjunto de prueba
test_size = int(len(Df_comparativa) * 0.2)  # 20% para prueba

# Calcular los índices
train_indices = list(range(len(Df_comparativa) - test_size))
test_indices = list(range(len(Df_comparativa) - test_size, len(Df_comparativa)))

# Dividir los datos en conjuntos de entrenamiento y prueba
train_data = Df_comparativa.iloc[train_indices]
test_data = Df_comparativa.iloc[test_indices]

# Definir el tamaño de la ventana para modelos comparativos
window_size = 12

# Definir y_test (valores reales de prueba)
y_test = test_data['ssn']

# Modelo de Media Móvil
y_train_ma = Df_comparativa['ssn'].iloc[train_indices]
y_test_ma = Df_comparativa['ssn'].iloc[test_indices]

# Media Móvil simple
y_pred_ma = y_test_ma.rolling(window=window_size).mean().shift(-window_size+1)
y_pred_ma = y_pred_ma.dropna()

# Ajustar y_test_ma para que coincida con la longitud de y_pred_ma
y_test_ma = y_test_ma.loc[y_pred_ma.index]
mse_ma = mean_squared_error(y_test_ma, y_pred_ma)
rmse_ma = mean_squared_error(y_test_ma, y_pred_ma, squared=False)
print(f'MSE del modelo de Media Móvil: {mse_ma}')
print(f'RMSE del modelo de Media Móvil: {rmse_ma}')

# Modelo de Promedio Exponencial
y_train_es = Df_comparativa['ssn'].iloc[train_indices]
y_test_es = Df_comparativa['ssn'].iloc[test_indices]

# Ajuste del modelo Exponential Smoothing
es_model = ExponentialSmoothing(y_train_es, trend='add', seasonal='add', seasonal_periods=window_size)
es_fit = es_model.fit()
y_pred_es = es_fit.forecast(len(y_test_es))

# Asegurarse de que y_test_es coincida con la longitud de y_pred_es
y_test_es = y_test_es.loc[y_pred_es.index]
mse_es = mean_squared_error(y_test_es, y_pred_es)
rmse_es = mean_squared_error(y_test_es, y_pred_es, squared=False)
print(f'MSE del modelo de Promedio Exponencial: {mse_es}')
print(f'RMSE del modelo de Promedio Exponencial: {rmse_es}')

# Modelo de Regresión Lineal
X_train = train_data[['ssn', 'smoothed_ssn']]  # Asegúrate de que estas columnas existan en tus datos
X_test = test_data[['ssn', 'smoothed_ssn']]
y_train = train_data['ssn']

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

# Reiniciar el índice de y_test para alinear con las predicciones
#y_test = y_test.reset_index(drop=True)
#mse_lr = mean_squared_error(y_test, y_pred_lr)
#print(f'MSE del modelo de Regresión Lineal: {mse_lr}')

# Se ajusta y_test_pred (de XGBoost) para que coincida con y_test
#  Se supone que y_test_pred son las predicciones obtenidas con el modelo XGBoost
#y_test_pred = XGB_model.predict(X_test)

#if len(y_test_pred) != len(y_test):
#y_test = y_test.iloc[-len(y_test_pred):]  # Se recorta y_test para que coincida en tamaño con y_test_pred
#mse_xgb = mean_squared_error(y_test, y_test_pred)

print(f'MSE del modelo XGBoost: {mse}')
print(f'RMSE del modelo XGBoost: {rmse}')



    

