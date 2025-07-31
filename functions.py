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

def normalize_minmax(value, min_val, max_val):
    """
    Aplica la normalización Min-Max a un valor.
    """
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)

# --- Carga de Datos, Entrenamiento y Preprocesamiento (Consolidado) ---
@st.cache_resource
def load_all_resources():
    """
    Carga los datasets, coeficientes Box-Cox, entrena el modelo de regresión,
    y prepara los DataFrames para la predicción y el cálculo de similitud.
    También calcula los parámetros de normalización.
    """
    # --- Carga de archivos ---
    try:
        df_model_ready = pd.read_csv('p_items/df_model_ready.csv')
    except FileNotFoundError:
        st.error("Error: df_model_ready.csv no encontrado.")
        st.stop()

    try:
        with open('p_items/boxcox_lambdas.json', 'r') as f:
            lambdas = json.load(f)
    except FileNotFoundError:
        st.error("Error: boxcox_lambdas.json no encontrado.")
        st.stop()

    try:
        df_original = pd.read_csv('p_items/p_laptops.csv')
    except FileNotFoundError:
        st.error("Error: p_laptops.csv no encontrado.")
        st.stop()

    categorical_cols = ['Company', 'TypeName', 'OS', 'PrimaryStorageType', 'SecondaryStorageType', 'CPU_company', 'GPU_company']

    # --- Preparación de datos y entrenamiento del modelo de predicción ---
    X_raw = df_model_ready.drop('Price_euros_BoxCox', axis=1)
    y = df_model_ready['Price_euros_BoxCox']
    X_processed = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

    model = LinearRegression()
    model.fit(X_processed, y)

    model_features_X = X_processed.columns.tolist()

    # --- Preparación de datos para el cálculo de similitud ---
    df_for_similarity = df_original.copy()
    df_for_similarity['ScreenPixels'] = df_for_similarity['ScreenW'] * df_for_similarity['ScreenH']

    numerical_cols_for_boxcox = [
        'Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage', 'ScreenPixels', 'Price_euros'
    ]
    
    normalization_params = {}
    for col in numerical_cols_for_boxcox:
        lambda_val = lambdas.get(col)
        if lambda_val is not None:
            # Primero aplicar Box-Cox
            if col == 'SecondaryStorage':
                df_for_similarity[col + '_BoxCox'] = df_for_similarity[col].apply(
                    lambda x: apply_boxcox(x, lambda_val, is_secondary_storage=True)
                )
            else:
                df_for_similarity[col + '_BoxCox'] = df_for_similarity[col].apply(
                    lambda x: apply_boxcox(x, lambda_val)
                )
            
            # Luego aplicar Min-Max Normalization y guardar los parámetros
            boxcox_col_name = col + '_BoxCox'
            min_val = df_for_similarity[boxcox_col_name].min()
            max_val = df_for_similarity[boxcox_col_name].max()
            normalization_params[boxcox_col_name] = {'min': min_val, 'max': max_val}
            df_for_similarity[boxcox_col_name] = df_for_similarity[boxcox_col_name].apply(
                lambda x: normalize_minmax(x, min_val, max_val)
            )

    binary_cols = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
    for col in binary_cols:
        df_for_similarity[col] = df_for_similarity[col].astype(int)

    # One-Hot Encoding
    df_for_similarity = pd.get_dummies(df_for_similarity, columns=categorical_cols, drop_first=True)

    # Definir el conjunto COMPLETO de características para la SIMILITUD (Box-Cox, normalizadas y categóricas)
    similarity_features = [col + '_BoxCox' for col in numerical_cols_for_boxcox] + binary_cols
    similarity_features.extend([col for col in df_for_similarity.columns if col.startswith(tuple(categorical_cols))])

    # Seleccionar y reindexar las columnas para que coincidan con similarity_features
    final_df_for_similarity = pd.DataFrame(0, index=df_for_similarity.index, columns=similarity_features)
    for col in final_df_for_similarity.columns:
        if col in df_for_similarity.columns:
            final_df_for_similarity[col] = df_for_similarity[col]
    
    # Calcular rangos para la puntuación de prioridad (usando datos originales no transformados)
    numerical_ranges = calculate_numerical_ranges(df_original)

    return df_original, lambdas, categorical_cols, model, model_features_X, similarity_features, final_df_for_similarity, numerical_ranges, normalization_params

# --- Calcular Rangos Numéricos (Caché) ---
@st.cache_data
def calculate_numerical_ranges(df_original):
    """
    Calcula los rangos (min, max, range) para las columnas numéricas del DataFrame original.
    """
    numerical_ranges = {}
    for col in ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage', 'ScreenW', 'ScreenH', 'Price_euros']:
        if col in df_original.columns:
            min_val = df_original[col].min()
            max_val = df_original[col].max()
            numerical_ranges[col] = {'min': min_val, 'max': max_val, 'range': max_val - min_val}
        else:
            if col == 'ScreenW' or col == 'ScreenH':
                numerical_ranges[col] = {'min': 1000, 'max': 4000, 'range': 3000} 
            elif col == 'Price_euros':
                numerical_ranges[col] = {'min': df_original['Price_euros'].min(), 'max': df_original['Price_euros'].max(), 'range': df_original['Price_euros'].max() - df_original['Price_euros'].min()}
    return numerical_ranges

# --- Función de Predicción ---
def predict_laptop_price(input_data, model, lambdas, categorical_cols, model_features_X, similarity_features, normalization_params):
    """
    Toma los datos de entrada del usuario, los transforma (Box-Cox y Normalización),
    predice el precio y prepara el DataFrame para el cálculo de similitud.
    Retorna el precio predicho en euros y el DataFrame de la laptop del usuario para similitud.
    """
    user_input_df = pd.DataFrame([input_data])
    
    # Aplicar Box-Cox
    user_input_df['Inches_BoxCox'] = apply_boxcox(user_input_df['Inches'].iloc[0], lambdas['Inches'])
    user_input_df['Ram_BoxCox'] = apply_boxcox(user_input_df['Ram'].iloc[0], lambdas['Ram'])
    user_input_df['Weight_BoxCox'] = apply_boxcox(user_input_df['Weight'].iloc[0], lambdas['Weight'])
    user_input_df['CPU_freq_BoxCox'] = apply_boxcox(user_input_df['CPU_freq'].iloc[0], lambdas['CPU_freq'])
    user_input_df['PrimaryStorage_BoxCox'] = apply_boxcox(user_input_df['PrimaryStorage'].iloc[0], lambdas['PrimaryStorage'])
    user_input_df['SecondaryStorage_BoxCox'] = apply_boxcox(user_input_df['SecondaryStorage'].iloc[0], lambdas['SecondaryStorage'], is_secondary_storage=True)
    user_input_df['ScreenPixels'] = user_input_df['ScreenW'] * user_input_df['ScreenH']
    user_input_df['ScreenPixels_BoxCox'] = apply_boxcox(user_input_df['ScreenPixels'].iloc[0], lambdas['ScreenPixels'])

    # Aplicar Normalización a las características numéricas Box-Cox
    numerical_boxcox_cols = ['Inches_BoxCox', 'Ram_BoxCox', 'Weight_BoxCox', 'CPU_freq_BoxCox',
                             'PrimaryStorage_BoxCox', 'SecondaryStorage_BoxCox', 'ScreenPixels_BoxCox']
    
    for col in numerical_boxcox_cols:
        params = normalization_params.get(col, {'min': 0, 'max': 1}) # Parámetros de normalización para similitud
        user_input_df[col + '_normalized'] = normalize_minmax(user_input_df[col].iloc[0], params['min'], params['max'])

    # Crear el DataFrame para la predicción (con los valores Box-Cox, no los normalizados)
    user_dummies = pd.get_dummies(user_input_df[categorical_cols], columns=categorical_cols, drop_first=True)
    final_input_df_for_prediction = pd.DataFrame(0, index=[0], columns=model_features_X)
    
    numerical_boxcox_cols_pred = [
        'Inches_BoxCox', 'Ram_BoxCox', 'Weight_BoxCox', 'CPU_freq_BoxCox',
        'PrimaryStorage_BoxCox', 'SecondaryStorage_BoxCox', 'ScreenPixels_BoxCox'
    ]
    binary_user_cols = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']

    for col in numerical_boxcox_cols_pred:
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

    # Crear el DataFrame de la laptop del usuario para el cálculo de similitud (con valores normalizados)
    user_laptop_for_similarity = pd.DataFrame(0, index=[0], columns=similarity_features)
    
    # Copiar las características numéricas Box-Cox normalizadas
    for col in numerical_boxcox_cols:
        normalized_col_name = col + '_normalized'
        if normalized_col_name in user_laptop_for_similarity.columns:
            user_laptop_for_similarity[normalized_col_name] = user_input_df[normalized_col_name].iloc[0]
            
    # Copiar las características binarias y categóricas
    for col in binary_user_cols:
        if col in user_laptop_for_similarity.columns:
            user_laptop_for_similarity[col] = user_input_df[col].iloc[0]
    
    for col in user_dummies.columns:
        if col in user_laptop_for_similarity.columns:
            user_laptop_for_similarity[col] = user_dummies[col].iloc[0]
    
    # Añadir el precio predicho, también transformado y normalizado
    price_boxcox_normalized = normalize_minmax(predicted_price_boxcox, 
                                             normalization_params['Price_euros_BoxCox']['min'],
                                             normalization_params['Price_euros_BoxCox']['max'])
    if 'Price_euros_BoxCox_normalized' in user_laptop_for_similarity.columns:
        user_laptop_for_similarity['Price_euros_BoxCox_normalized'] = price_boxcox_normalized

    return predicted_price_euros, user_laptop_for_similarity

# --- Función de Recomendación ---
def get_laptop_recommendations(user_laptop_for_similarity, user_original_data, selected_priorities, df_original, df_for_similarity, numerical_ranges, normalization_params):
    """
    Genera recomendaciones de laptops basadas en la similitud del coseno y prioridades del usuario.
    Retorna un DataFrame con las laptops recomendadas.
    """
    # El DataFrame df_for_similarity ya está normalizado, así que cosine_similarity es más preciso ahora.
    similarities = cosine_similarity(user_laptop_for_similarity, df_for_similarity)
    
    num_to_consider = min(50, len(df_original)) 
    initial_similar_indices = similarities.flatten().argsort()[-(num_to_consider + 1):-1][::-1]
    
    temp_df = df_original.iloc[initial_similar_indices].copy()
    temp_df['original_similarity'] = similarities.flatten()[initial_similar_indices]
    temp_df['priority_score'] = 0.0 # Inicializar la puntuación de prioridad

    bonus_weights = {0: 3.0, 1: 2.0, 2: 1.0} 

    for i, feature_name in enumerate(selected_priorities):
        if feature_name in user_original_data:
            user_value = user_original_data[feature_name]
            
            if feature_name in ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage', 'ScreenW', 'ScreenH', 'Price_euros']:
                # Para características numéricas, usamos la distancia en el espacio normalizado.
                
                # Primero, transformamos el valor del usuario a Box-Cox y luego lo normalizamos
                user_boxcox_val = apply_boxcox(user_value, lambdas[feature_name], is_secondary_storage=(feature_name == 'SecondaryStorage'))
                params = normalization_params.get(feature_name + '_BoxCox')
                if params:
                    user_normalized_val = normalize_minmax(user_boxcox_val, params['min'], params['max'])
                    
                    # Buscamos la columna normalizada correspondiente en el df temporal
                    normalized_col = feature_name + '_BoxCox_normalized'
                    if normalized_col in temp_df.columns:
                        proximity_score = 1 - abs(user_normalized_val - temp_df[normalized_col])
                        temp_df['priority_score'] += (proximity_score * bonus_weights[i])
                    
            elif feature_name in ['Touchscreen', 'IPSpanel', 'RetinaDisplay']:
                # Para características binarias, usamos la coincidencia exacta
                temp_df['priority_score'] += temp_df.apply(
                    lambda row: bonus_weights[i] if (user_value == row[feature_name]) else 0, axis=1
                )
            else:
                # Para todas las demás características (categóricas), coincidencia exacta
                temp_df['priority_score'] += temp_df.apply(
                    lambda row: bonus_weights[i] if (user_value == row[feature_name]) else 0, axis=1
                )
    
    temp_df['final_score'] = temp_df['original_similarity'] + temp_df['priority_score']

    final_recommended_laptops = temp_df.sort_values(by='final_score', ascending=False).head(5)

    return final_recommended_laptops