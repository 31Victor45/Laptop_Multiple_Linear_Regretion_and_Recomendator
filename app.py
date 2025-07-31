import streamlit as st
import pandas as pd
import numpy as np
# Importar todas las funciones necesarias desde el archivo functions.py
from functions import load_all_resources, calculate_numerical_ranges, predict_laptop_price, get_laptop_recommendations

# --- Configuración de la página de Streamlit (DEBE SER LO PRIMERO) ---
st.set_page_config(layout="wide")

# --- Carga de Datos y Modelo (Caché) ---
# Se utiliza una única función para cargar todos los recursos una sola vez.
(df_original, lambdas, categorical_cols, model, 
 model_features_X, similarity_features, 
 df_for_similarity, numerical_ranges, 
 normalization_params) = load_all_resources()

# --- 4. Interfaz de Usuario en Streamlit ---
st.title(" Predicción del Precio de Laptops 💻")
st.write("Ingresa las características de la laptop para predecir su precio en euros.")

# Inicializar variables de session_state si no existen
if 'final_input_df_for_prediction' not in st.session_state: 
    st.session_state.final_input_df_for_prediction = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'user_laptop_for_similarity' not in st.session_state: 
    st.session_state.user_laptop_for_similarity = None
if 'predicted_price_euros' not in st.session_state: 
    st.session_state.predicted_price_euros = None
if 'predicted_price_display' not in st.session_state: 
    st.session_state.predicted_price_display = None


# Diseño de columnas para una mejor organización
col1, col2, col3 = st.columns(3)

with col1:
    # 
    st.image("p_items/img/img_1.png", width=200) # Imagen para Especificaciones Generales
    st.header("Especificaciones Generales")
    company = st.selectbox("Compañía", df_original['Company'].unique())
    typename = st.selectbox("Tipo de Laptop", df_original['TypeName'].unique())
    os = st.selectbox("Sistema Operativo", df_original['OS'].unique())
    inches = st.number_input("Tamaño de Pantalla (pulgadas)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    ram = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8, step=2)
    weight = st.number_input("Peso (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

with col2:
    # 
    st.image("p_items/img/img_2.png", width=200) # Imagen para Almacenamiento y Pantalla
    st.header("Almacenamiento y Pantalla")
    primary_storage = st.number_input("Almacenamiento Principal (GB)", min_value=64, max_value=2048, value=256, step=64)
    primary_storage_type = st.selectbox("Tipo de Almacenamiento Principal", df_original['PrimaryStorageType'].unique())
    secondary_storage = st.number_input("Almacenamiento Secundario (GB, 0 si no aplica)", min_value=0, max_value=2048, value=0, step=64)
    # Filtrar 'No Aplica' para la opción por defecto si no existe en el original, o usar el original
    sec_storage_types = ['No Aplica'] + [s for s in df_original['SecondaryStorageType'].unique() if pd.notna(s)]
    secondary_storage_type = st.selectbox("Tipo de Almacenamiento Secundario", sec_storage_types)

    st.subheader("Características de Pantalla")
    touchscreen_str = st.radio("Pantalla Táctil", ("No", "Sí"))
    ips_panel_str = st.radio("Panel IPS", ("No", "Sí"))
    retina_display_str = st.radio("Pantalla Retina", ("No", "Sí"))
    screen_w = st.number_input("Ancho de Pantalla (píxeles)", min_value=1000, max_value=4000, value=1920, step=10)
    screen_h = st.number_input("Alto de Pantalla (píxeles)", min_value=700, max_value=3000, value=1080, step=10)

with col3:
    # 
    st.image("p_items/img/img_3.png", width=200) # Imagen para Procesadores
    st.header("Procesadores")
    cpu_company = st.selectbox("Compañía CPU", df_original['CPU_company'].unique())
    cpu_freq = st.number_input("Frecuencia CPU (GHz)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    gpu_company = st.selectbox("Compañía GPU", df_original['GPU_company'].unique())

# --- Botón de Predicción ---
st.write("---")
if st.button("Predecir Precio"):
    # Crear un diccionario con los valores de entrada del usuario
    input_data = {
        'Company': company, 'TypeName': typename, 'OS': os,
        'PrimaryStorageType': primary_storage_type,
        'SecondaryStorageType': secondary_storage_type if secondary_storage_type != 'No Aplica' else np.nan,
        'CPU_company': cpu_company, 'GPU_company': gpu_company,
        'Inches': inches, 'Ram': ram, 'Weight': weight, 'CPU_freq': cpu_freq,
        'PrimaryStorage': primary_storage, 'SecondaryStorage': secondary_storage,
        'Touchscreen': 1 if touchscreen_str == "Sí" else 0,
        'IPSpanel': 1 if ips_panel_str == "Sí" else 0,
        'RetinaDisplay': 1 if retina_display_str == "Sí" else 0,
        'ScreenW': screen_w, 'ScreenH': screen_h
    }

    try:
        # Llamar a la función de predicción desde functions.py
        # Se pasa el nuevo parámetro `normalization_params`
        predicted_price_euros, user_laptop_for_similarity = predict_laptop_price(
            input_data, model, lambdas, categorical_cols, model_features_X, similarity_features, normalization_params
        )

        # Almacenar en session_state para persistencia
        st.session_state.user_laptop_for_similarity = user_laptop_for_similarity
        st.session_state.input_data = input_data
        st.session_state.predicted_price_euros = predicted_price_euros
        st.session_state.predicted_price_display = f"El precio predicho para esta laptop es: **€{predicted_price_euros:,.2f}**"

        st.success(st.session_state.predicted_price_display)
    except Exception as e:
        st.error(f"Error al realizar la predicción. Detalle: {e}")

# --- Mostrar el precio predicho de forma persistente si está disponible ---
if st.session_state.predicted_price_display:
    st.markdown(st.session_state.predicted_price_display)

# --- Sección de Recomendación en Sidebar ---
st.sidebar.header("Recomendación de Laptops")
st.sidebar.image("p_items/img/img_4.png", width=320) # Imagen para la sección de recomendación
st.sidebar.write("Encuentra laptops similares a la que has configurado.")

# Lista de todas las características que pueden ser priorizadas
prioritizable_features = [
    'Company', 'TypeName', 'OS', 'Inches', 'Ram', 'Weight',
    'PrimaryStorage', 'PrimaryStorageType', 'SecondaryStorage', 'SecondaryStorageType',
    'ScreenW', 'ScreenH', 'Touchscreen', 'IPSpanel', 'RetinaDisplay',
    'CPU_company', 'CPU_freq', 'GPU_company', 'Price_euros' 
]

# Verificar si se ha configurado una laptop para habilitar la selección de prioridades
if st.session_state.user_laptop_for_similarity is not None and st.session_state.input_data is not None:
    st.sidebar.subheader("Prioriza Características:")
    priority1 = st.sidebar.selectbox("1ra Característica más importante", ['Seleccionar'] + prioritizable_features, key='p1')
    priority2 = st.sidebar.selectbox("2da Característica más importante", ['Seleccionar'] + prioritizable_features, key='p2')
    priority3 = st.sidebar.selectbox("3ra Característica más importante", ['Seleccionar'] + prioritizable_features, key='p3')

    if st.sidebar.button("Buscar Laptops Recomendadas"):
        selected_priorities = []
        if priority1 != 'Seleccionar': selected_priorities.append(priority1)
        if priority2 != 'Seleccionar': selected_priorities.append(priority2)
        if priority3 != 'Seleccionar': selected_priorities.append(priority3)

        if not selected_priorities:
            st.warning("Por favor, selecciona al menos una característica para priorizar.")
            with st.spinner('Buscando laptops similares sin prioridades específicas...'):
                final_recommended_laptops = get_laptop_recommendations(
                    st.session_state.user_laptop_for_similarity,
                    st.session_state.input_data,
                    [], # Lista vacía si no hay prioridades seleccionadas
                    df_original,
                    df_for_similarity,
                    numerical_ranges,
                    normalization_params
                )
            st.subheader("Las 5 Laptops Más Similares (Sin Priorización):")
        else:
            with st.spinner('Buscando laptops similares con tus prioridades...'):
                final_recommended_laptops = get_laptop_recommendations(
                    st.session_state.user_laptop_for_similarity,
                    st.session_state.input_data,
                    selected_priorities,
                    df_original,
                    df_for_similarity,
                    numerical_ranges,
                    normalization_params
                )
            st.subheader("Las 5 Laptops Más Similares (Priorizadas):")
        
        # Columnas a mostrar en la tabla de resultados
        display_cols = [
            'Company', 'TypeName', 'OS', 'Inches', 'Ram', 'Weight',
            'PrimaryStorage', 'PrimaryStorageType', 'SecondaryStorage', 'SecondaryStorageType',
            'ScreenW', 'ScreenH', 'Touchscreen', 'IPSpanel', 'RetinaDisplay',
            'CPU_company', 'CPU_freq', 'GPU_company', 'Price_euros'
        ]
        display_cols_present = [col for col in display_cols if col in final_recommended_laptops.columns]
        st.dataframe(final_recommended_laptops[display_cols_present].reset_index(drop=True).style.format({"Price_euros": "€{:,.2f}"}))
else:
    st.sidebar.warning("Por favor, predice un precio primero para configurar la laptop de referencia y buscar recomendaciones.")

st.markdown("""
---
**Nota:** Este modelo predice el precio en euros basándose en las características ingresadas.
Las transformaciones Box-Cox y el One-Hot Encoding se aplican internamente para la predicción.
""")
