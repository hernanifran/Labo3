import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
import optuna

# Cargar los datos
df = pd.read_csv('C:/Users/Hernán Ifrán/Downloads/df_prediccion_all.csv')  # Cambio de dataset final con descripciones
productosPredecir = pd.read_csv('C:/Users/Hernán Ifrán/Downloads/productos_a_predecir.txt', sep='\t')

# Convertir el periodo a formato datetime
df['periodo'] = pd.to_datetime(df['periodo'], format='%Y%m')

# Agregar los datos por periodo y product_id para obtener la serie temporal
ts = df.groupby(['periodo', 'product_id'])['tn'].sum().reset_index()

# Asegurarse de que las columnas tengan el mismo tipo y formato
ts['product_id'] = ts['product_id'].astype(int)
ts['periodo'] = pd.to_datetime(ts['periodo'])

# Crear características adicionales cat1    
ts['cat1'] = ts['product_id'] % 2 

# Codificar la característica cat1 utilizando One-Hot Encoding
ts = pd.get_dummies(ts, columns=['cat1'], prefix='cat1', drop_first=True)

# Agregar lags a los datos
lags = 3  # Número de lags a incluir
for lag in range(1, lags + 1):
    ts[f'tn_lag_{lag}'] = ts.groupby('product_id')['tn'].shift(lag)

# Calcular la media de las ventas para cada producto
ts['tn_mean'] = ts.groupby('product_id')['tn'].transform('mean')

# Eliminar filas con valores NaN en la variable objetivo
ts.dropna(subset=['tn'], inplace=True)

# Crear características adicionales si es necesario (Ejemplo: características temporales)
ts['year'] = ts['periodo'].dt.year
ts['month'] = ts['periodo'].dt.month

# Generar características ARIMA para cada producto
def generate_arima_features(data, p=1, d=1, q=1):
    arima_preds = []
    for product_id in data['product_id'].unique():
        product_data = data[data['product_id'] == product_id].copy()
        model = ARIMA(product_data['tn'], order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=2)  # Predice los próximos dos periodos
        arima_preds.append(forecast.values)
    
    arima_preds = np.array(arima_preds)
    return arima_preds

# Aplicar la función para generar características ARIMA
arima_features = generate_arima_features(ts)
ts['arima_pred_1'] = np.nan
ts['arima_pred_2'] = np.nan

# Agregar predicciones ARIMA al DataFrame ts
for idx, product_id in enumerate(ts['product_id'].unique()):
    ts.loc[ts['product_id'] == product_id, 'arima_pred_1'] = arima_features[idx, 0]
    ts.loc[ts['product_id'] == product_id, 'arima_pred_2'] = arima_features[idx, 1]

# Obtener la lista de productos únicos a predecir
product_ids = productosPredecir['product_id'].unique()

# Crear conjunto de entrenamiento y objetivo
feature_columns = ['product_id', 'year', 'month', 'tn_mean', 'cat1_1', 'arima_pred_1', 'arima_pred_2'] + [f'tn_lag_{lag}' for lag in range(1, lags + 1)]
X = ts[feature_columns].astype(float)
y = ts['tn'].shift(-2)

# Eliminar filas con valores NaN en el conjunto de datos
y.fillna(0, inplace=True)

# Calcular los pesos en función de tn
weights = ts['tn'] / ts['tn'].sum()

# Imputar valores faltantes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Utilizar TimeSeriesSplit para validación cruzada en series temporales
tscv = TimeSeriesSplit(n_splits=5)

# Definir la función objetivo para Optuna
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
    }

    model = lgb.LGBMRegressor(**params)
    
    # Usar TimeSeriesSplit para validación cruzada
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train = weights[train_index]
        model.fit(X_train, y_train, sample_weight=weights_train)
        preds = model.predict(X_test)
        score = mean_squared_error(y_test, preds)
        scores.append(score)
    
    return np.mean(scores)

# Crear el estudio de Optuna y optimizar los hiperparámetros
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

# Obtener el mejor conjunto de hiperparámetros
best_params = study.best_params

# Entrenar el modelo final con los mejores hiperparámetros
best_lgbm = lgb.LGBMRegressor(**best_params)
best_lgbm.fit(X, y, sample_weight=weights)

# Calcular la importancia de las características
feature_importances = best_lgbm.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_columns, 'importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Imprimir la importancia de las características
print(feature_importance_df)

# Calcular el promedio global de tn
global_mean_tn = ts['tn'].mean()

# Realizar predicciones para los productos a predecir
results = []
for product_id in tqdm(product_ids, desc="Predicting with Optimized LGBM"):
    product_data = ts[ts['product_id'] == product_id].copy()
    if not product_data.empty:
        # Predicción para el último periodo disponible + 2 meses
        last_period = product_data['periodo'].max()
        next_period = last_period + pd.DateOffset(months=2)
        next_data = pd.DataFrame({
            'product_id': [product_id],
            'year': [next_period.year],
            'month': [next_period.month],
            'tn_mean': [product_data['tn_mean'].iloc[-1]],
            'cat1_1': [product_data['cat1_1'].iloc[-1]],
            'tn_lag_1': [product_data['tn_lag_1'].iloc[-1]],
            'tn_lag_2': [product_data['tn_lag_2'].iloc[-1]],
            'tn_lag_3': [product_data['tn_lag_3'].iloc[-1]],
            'arima_pred_1': [product_data['arima_pred_1'].iloc[-1]],
            'arima_pred_2': [product_data['arima_pred_2'].iloc[-1]]
        })
        
        # Ordenar las columnas de next_data de acuerdo a feature_columns
        next_data = next_data[feature_columns]

        # Imputar valores faltantes en la predicción
        next_data_imputed = imputer.transform(next_data)
        
        # Realizar la predicción con LightGBM
        lgbm_pred = best_lgbm.predict(next_data_imputed)[0]
        
        # Predicción final promedio entre ARIMA y LightGBM
        final_pred = (lgbm_pred + product_data['arima_pred_2'].iloc[-1]) / 2
        results.append({
            'product_id': product_id,
            'predicted_tn': final_pred
        })
    
    else:
        # Calcular el promedio de tn para el producto si existen datos históricos
        product_mean_tn = ts[ts['product_id'] == product_id]['tn'].mean()
        if not np.isnan(product_mean_tn):
            results.append({'product_id': product_id, 'predicted_tn': product_mean_tn})
        else:
            # Si no hay datos históricos, usar el promedio global
            results.append({'product_id': product_id, 'predicted_tn': global_mean_tn})

# Convertir los resultados a un DataFrame
results_df = pd.DataFrame(results)

# Asegurarse de que el DataFrame resultante tiene las columnas product_id y predicted_tn
results_df = results_df[['product_id', 'predicted_tn']]

# Exportar a un archivo CSV con las columnas product_id y predicted_tn
results_df.to_csv('C:/Users/Hernán Ifrán/Downloads/ensemble_predictions.csv', index=False)

print("Predicciones exportadas a 'ensemble_predictions.csv'")

# Calcular las métricas de error
mae = mean_absolute_error(y, best_lgbm.predict(X))
rmse = mean_squared_error(y, best_lgbm.predict(X), squared=False)
r2 = r2_score(y, best_lgbm.predict(X))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
