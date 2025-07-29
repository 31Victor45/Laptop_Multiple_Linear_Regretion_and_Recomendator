import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from scipy.special import inv_boxcox
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st # Necesario para st.cache_data y st.cache_resource, y st.error/st.stop

# --- Funciones de Ayuda ---
def apply_boxcox(value, lambda_val, is_secondary_storage=False):
    """
    Aplica la transformación Box-Cox a un valor.
    Maneja valores <= 0 sumando una pequeña constante,
    o específicamente transforma 0 a 1 para SecondaryStorage si se indica.
    """
    # Asegurarse de que el valor sea un tipo numérico (float o int)
    value = float(value) 

    if is_secondary_storage and value == 0:
        # Si es SecondaryStorage y el valor es 0, transformarlo a 1 antes de Box-Cox
        # Esto replica el preprocesamiento original donde 0 se convirtió a 1.
        value = 1
    elif value <= 0:
        # Para otras columnas, sumar una pequeña constante para que sea positivo y no cero
        value = value + 1e-6

    if lambda_val == 0:
        return np.log(value)
    else:
        return (value**lambda_val - 1) / lambda_val

def inverse_boxcox(value, lambda_val):
    """
    Revierte la transformación Box-Cox a un valor.
    """
    return inv_boxcox(value, lambda_val)

# --- Carga de Datos y Entrenamiento del Modelo ---
@st.cache_resource # Cachea el modelo y objetos grandes para evitar recargas
def load_data_and_train_model():
    """
    Carga los datasets, coeficientes Box-Cox, define columnas categóricas,
    entrena el modelo de regresión lineal y prepara las listas de características.
    """
    try:
        df_model_ready = pd.read_csv('p_items/df_model_ready.csv')
    except FileNotFoundError:
        st.error("Error: df_model_ready.csv no encontrado. Asegúrate de que el archivo esté en la ruta correcta: 'p_items/df_model_ready.csv'")
        st.stop()

    try:
        with open('p_items/boxcox_lambdas.json', 'r') as f:
            lambdas = json.load(f)
    except FileNotFoundError:
        st.error("Error: boxcox_lambdas.json no encontrado. Asegúrate de que el archivo esté en la ruta correcta: 'p_items/boxcox_lambdas.json'")
        st.stop()

    try:
        df_original = pd.read_csv('p_items/p_laptops.csv')
    except FileNotFoundError:
        st.error("Error: p_laptops.csv no encontrado. Asegúrate de que el archivo esté en la ruta correcta: 'p_items/p_laptops.csv'")
        st.stop()

    categorical_cols = ['Company', 'TypeName', 'OS', 'PrimaryStorageType', 'SecondaryStorageType', 'CPU_company', 'GPU_company']

    # Preparación de datos para el entrenamiento del modelo
    X_raw = df_model_ready.drop('Price_euros_BoxCox', axis=1)
    y = df_model_ready['Price_euros_BoxCox']
    X_processed = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

    # Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X_processed, y)

    # Guardar la lista de columnas que el modelo espera para las CARACTERÍSTICAS (X)
    model_features_X = X_processed.columns.tolist()
    # Definir el conjunto COMPLETO de características para la SIMILITUD (X + y)
    similarity_features = model_features_X + ['Price_euros_BoxCox']

    return df_original, lambdas, categorical_cols, model, model_features_X, similarity_features

# --- Preparar DataFrame para Cálculo de Similitud (Caché) ---
@st.cache_data # Cachea el DataFrame procesado para la similitud
def prepare_df_for_similarity(df, lambdas, categorical_cols, similarity_features):
    """
    Prepara el DataFrame original aplicando Box-Cox y One-Hot Encoding
    para que sea compatible con el espacio de características del modelo de similitud.
    """
    df_processed_for_similarity = df.copy()

    # Calcular ScreenPixels
    df_processed_for_similarity['ScreenPixels'] = df_processed_for_similarity['ScreenW'] * df_processed_for_similarity['ScreenH']

    # Aplicar Box-Cox a las columnas numéricas (incluyendo Price_euros)
    numerical_cols_for_boxcox = [
        'Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage', 'ScreenPixels', 'Price_euros'
    ]
    
    for col in numerical_cols_for_boxcox:
        lambda_val = lambdas.get(col)
        if lambda_val is not None:
            if col == 'SecondaryStorage':
                df_processed_for_similarity[col + '_BoxCox'] = df_processed_for_similarity[col].apply(
                    lambda x: apply_boxcox(x, lambda_val, is_secondary_storage=True)
                )
            elif col == 'Price_euros':
                df_processed_for_similarity[col + '_BoxCox'] = df_processed_for_similarity[col].apply(
                    lambda x: apply_boxcox(x, lambda_val)
                )
            else:
                df_processed_for_similarity[col + '_BoxCox'] = df_processed_for_similarity[col].apply(
                    lambda x: apply_boxcox(x, lambda_val)
                )
        # No es necesario un st.warning aquí, ya que se maneja la ausencia de lambda
        # y la columna simplemente no se transforma, lo cual es el comportamiento deseado.

    # Convertir variables binarias (ya deberían ser 0/1, pero asegurar el tipo)
    binary_cols = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
    for col in binary_cols:
        df_processed_for_similarity[col] = df_processed_for_similarity[col].astype(int)

    # Aplicar One-Hot Encoding a las columnas categóricas
    df_processed_for_similarity = pd.get_dummies(df_processed_for_similarity, columns=categorical_cols, drop_first=True)

    # Seleccionar y reindexar las columnas para que coincidan con similarity_features
    final_df_for_similarity = pd.DataFrame(0, index=df_processed_for_similarity.index, columns=similarity_features)
    
    for col in final_df_for_similarity.columns:
        if col in df_processed_for_similarity.columns:
            final_df_for_similarity[col] = df_processed_for_similarity[col]

    return final_df_for_similarity

# --- Calcular Rangos Numéricos (Caché) ---
@st.cache_data # Cachea los rangos numéricos
def calculate_numerical_ranges(df_original):
    """
    Calcula los rangos (min, max, range) para las columnas numéricas del DataFrame original,
    usados para la normalización en la puntuación de proximidad.
    """
    numerical_ranges = {}
    for col in ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage', 'ScreenW', 'ScreenH', 'Price_euros']:
        if col in df_original.columns:
            min_val = df_original[col].min()
            max_val = df_original[col].max()
            numerical_ranges[col] = {'min': min_val, 'max': max_val, 'range': max_val - min_val}
        else:
            # Valores por defecto si la columna no existe o para columnas derivadas
            if col == 'ScreenW' or col == 'ScreenH':
                numerical_ranges[col] = {'min': 1000, 'max': 4000, 'range': 3000} 
            elif col == 'Price_euros':
                 numerical_ranges[col] = {'min': df_original['Price_euros'].min(), 'max': df_original['Price_euros'].max(), 'range': df_original['Price_euros'].max() - df_original['Price_euros'].min()}
    return numerical_ranges

# --- Función de Predicción ---
def predict_laptop_price(input_data, model, lambdas, categorical_cols, model_features_X, similarity_features):
    """
    Toma los datos de entrada del usuario, los transforma, predice el precio
    y prepara el DataFrame para el cálculo de similitud.
    Retorna el precio predicho en euros y el DataFrame de la laptop del usuario para similitud.
    """
    user_input_df = pd.DataFrame([input_data])

    user_input_df['Inches_BoxCox'] = apply_boxcox(user_input_df['Inches'].iloc[0], lambdas['Inches'])
    user_input_df['Ram_BoxCox'] = apply_boxcox(user_input_df['Ram'].iloc[0], lambdas['Ram'])
    user_input_df['Weight_BoxCox'] = apply_boxcox(user_input_df['Weight'].iloc[0], lambdas['Weight'])
    user_input_df['CPU_freq_BoxCox'] = apply_boxcox(user_input_df['CPU_freq'].iloc[0], lambdas['CPU_freq'])
    user_input_df['PrimaryStorage_BoxCox'] = apply_boxcox(user_input_df['PrimaryStorage'].iloc[0], lambdas['PrimaryStorage'])
    user_input_df['SecondaryStorage_BoxCox'] = apply_boxcox(user_input_df['SecondaryStorage'].iloc[0], lambdas['SecondaryStorage'], is_secondary_storage=True)

    user_input_df['ScreenPixels'] = user_input_df['ScreenW'] * user_input_df['ScreenH']
    user_input_df['ScreenPixels_BoxCox'] = apply_boxcox(user_input_df['ScreenPixels'].iloc[0], lambdas['ScreenPixels'])

    user_dummies = pd.get_dummies(user_input_df[categorical_cols], columns=categorical_cols, drop_first=True)

    final_input_df_for_prediction = pd.DataFrame(0, index=[0], columns=model_features_X)
    
    numerical_boxcox_cols = ['Inches_BoxCox', 'Ram_BoxCox', 'Weight_BoxCox', 'CPU_freq_BoxCox',
                             'PrimaryStorage_BoxCox', 'SecondaryStorage_BoxCox', 'ScreenPixels_BoxCox']
    binary_user_cols = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']

    for col in numerical_boxcox_cols:
        if col in final_input_df_for_prediction.columns:
             final_input_df_for_prediction[col] = user_input_df[col].iloc[0]

    for col in binary_user_cols:
        if col in final_input_df_for_prediction.columns:
            final_input_df_for_prediction[col] = user_input_df[col].iloc[0]

    for col in user_dummies.columns:
        if col in final_input_df_for_prediction.columns:
            final_input_df_for_prediction[col] = user_dummies[col].iloc[0]

    predicted_price_boxcox = model.predict(final_input_df_for_prediction)[0]
    predicted_price_euros = inverse_boxcox(predicted_price_boxcox, lambdas['Price_euros'])

    # Crear el DataFrame de la laptop del usuario para el cálculo de similitud
    user_laptop_for_similarity = pd.DataFrame(0, index=[0], columns=similarity_features)
    # Copiar las características X de final_input_df_for_prediction
    for col in model_features_X:
        if col in user_laptop_for_similarity.columns:
            user_laptop_for_similarity[col] = final_input_df_for_prediction[col].iloc[0]
    
    # Añadir el precio predicho en escala Box-Cox
    if 'Price_euros_BoxCox' in user_laptop_for_similarity.columns:
        user_laptop_for_similarity['Price_euros_BoxCox'] = predicted_price_boxcox

    return predicted_price_euros, user_laptop_for_similarity

# --- Función de Recomendación ---
def get_laptop_recommendations(user_laptop_for_similarity, user_original_data, selected_priorities, df_original, df_for_similarity, numerical_ranges, lambdas):
    """
    Genera recomendaciones de laptops basadas en la similitud del coseno y prioridades del usuario.
    Retorna un DataFrame con las laptops recomendadas.
    """
    similarities = cosine_similarity(user_laptop_for_similarity, df_for_similarity)
    
    num_to_consider = min(50, len(df_original)) 
    # Obtener los índices de las laptops inicialmente más similares (excluyendo posiblemente la propia laptop configurada si está en el dataset)
    initial_similar_indices = similarities.flatten().argsort()[-(num_to_consider + 1):-1][::-1]
    
    temp_df = df_original.iloc[initial_similar_indices].copy()
    temp_df['original_similarity'] = similarities.flatten()[initial_similar_indices]
    temp_df['priority_score'] = 0.0 # Inicializar la puntuación de prioridad

    # Definir pesos de bonificación para cada nivel de prioridad
    bonus_weights = {0: 3.0, 1: 2.0, 2: 1.0} 

    for i, feature_name in enumerate(selected_priorities):
        if feature_name in user_original_data:
            user_value = user_original_data[feature_name]
            
            # Manejar diferentes tipos de datos y casos especiales para la comparación
            if feature_name == 'SecondaryStorageType':
                temp_df['priority_score'] += temp_df.apply(
                    lambda row: bonus_weights[i] if (
                        (pd.isna(user_value) and pd.isna(row[feature_name])) or
                        (str(user_value) == str(row[feature_name])) # Comparar como cadenas para no-NaN
                    ) else 0, axis=1
                )
            elif feature_name == 'SecondaryStorage':
                 temp_df['priority_score'] += temp_df.apply(
                    lambda row: bonus_weights[i] if (
                        (user_value == 0 and row[feature_name] == 0) or
                        (user_value > 0 and row[feature_name] > 0 and user_value == row[feature_name])
                    ) else 0, axis=1
                )
            elif feature_name in ['Touchscreen', 'IPSpanel', 'RetinaDisplay']:
                temp_df['priority_score'] += temp_df.apply(
                    lambda row: bonus_weights[i] if (user_value == row[feature_name]) else 0, axis=1
                )
            elif feature_name == 'Price_euros':
                # Para el precio, usar una puntuación de proximidad continua
                user_predicted_price = inverse_boxcox(user_laptop_for_similarity['Price_euros_BoxCox'].iloc[0], lambdas['Price_euros'])
                if numerical_ranges['Price_euros']['range'] > 0:
                    price_diff = abs(user_predicted_price - temp_df[feature_name])
                    proximity_score = 1 - (price_diff / numerical_ranges['Price_euros']['range'])
                    proximity_score = proximity_score.clip(lower=0) 
                    temp_df['priority_score'] += (proximity_score * bonus_weights[i])
                else: # Manejar caso donde todos los precios son iguales (rango es 0)
                    temp_df['priority_score'] += temp_df.apply(
                        lambda row: bonus_weights[i] if (user_predicted_price == row[feature_name]) else 0, axis=1
                    )
            elif feature_name in ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'ScreenW', 'ScreenH']:
                # Para otras características numéricas, también usar puntuación de proximidad continua
                if numerical_ranges[feature_name]['range'] > 0:
                    feature_diff = abs(user_value - temp_df[feature_name])
                    proximity_score = 1 - (feature_diff / numerical_ranges[feature_name]['range'])
                    proximity_score = proximity_score.clip(lower=0)
                    temp_df['priority_score'] += (proximity_score * bonus_weights[i])
                else: # Manejar caso donde todos los valores para esta característica son iguales
                    temp_df['priority_score'] += temp_df.apply(
                        lambda row: bonus_weights[i] if (user_value == row[feature_name]) else 0, axis=1
                    )
            else:
                # Para todas las demás características (categóricas), coincidencia exacta
                temp_df['priority_score'] += temp_df.apply(
                    lambda row: bonus_weights[i] if (user_value == row[feature_name]) else 0, axis=1
                )
    
    # Combinar la similitud original con la puntuación de prioridad
    temp_df['final_score'] = temp_df['original_similarity'] + temp_df['priority_score']

    # Ordenar por final_score y obtener las 5 mejores
    final_recommended_laptops = temp_df.sort_values(by='final_score', ascending=False).head(5)

    return final_recommended_laptops
