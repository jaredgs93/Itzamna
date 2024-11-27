import streamlit as st
import requests

# Configurar diseño de la página a "wide"
st.set_page_config(layout="wide")


# Definir los endpoints
endpoint_antecedentes = "http://127.0.0.1:8000/antecedents"
endpoint_consecuentes = "http://127.0.0.1:8000/consequents"
endponint_reglas = "http://127.0.0.1:8000/create_rule/"

# Obtener los datos desde los endpoints
@st.cache_data
def obtener_datos(endpoint):
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error al obtener datos del endpoint")
        return []

# Cargar datos de antecedentes y consecuentes
datos_antecedentes = obtener_datos(endpoint_antecedentes)
datos_consecuentes = obtener_datos(endpoint_consecuentes)

# Crear diccionarios para mapear 'variable_description' a datos requeridos en orden alfabético
opciones_antecedentes = {
    item['variable_description']: {
        "fuzzy_sets": item['fuzzy_sets'],
        "fuzzy_sets_es": item['fuzzy_sets_es'],
        "variable_name_es": item['variable_name_es']
    } for item in sorted(datos_antecedentes, key=lambda x: x['variable_description'])
}
opciones_consecuentes = {
    item['variable_description']: {
        "fuzzy_sets": item['fuzzy_sets'],
        "fuzzy_sets_es": item['fuzzy_sets_es'],
        "variable_name_es": item['variable_name_es']
    } for item in sorted(datos_consecuentes, key=lambda x: x['variable_description'])
}

# Título de la interfaz
st.title("Creation of Soft Skills Assessment Rules")

# Función para construir el antecedente
def construir_antecedente():
    antecedente = ""
    condiciones = [
        (descripcion_1, es_o_no_1, conjunto_1, operador_1),
        (descripcion_2, es_o_no_2, conjunto_2, operador_2) if operador_1 else None,
        (descripcion_3, es_o_no_3, conjunto_3, None) if operador_2 else None
    ]
    
    for i, condicion in enumerate(condiciones):
        if condicion:
            variable, es_o_no, conjunto, operador = condicion
            if variable and conjunto:
                # Obtener valores en español para la variable y conjunto borroso
                variable_es = opciones_antecedentes[variable]["variable_name_es"]
                index_conjunto = opciones_antecedentes[variable]["fuzzy_sets"].index(conjunto)
                conjunto_es = opciones_antecedentes[variable]["fuzzy_sets_es"][index_conjunto]
                
                # Agregar "NOT" si es "IS NOT"
                condicion_texto = f"{'NOT ' if es_o_no == 'IS NOT' else ''}{variable_es}[{conjunto_es}]"
                antecedente += condicion_texto
                if operador:  # Agregar el operador (AND/OR) si no es el último
                    antecedente += f" {operador} "
                    
    return 'IF '+ antecedente

# Antecedente 1
st.subheader("Antecedent 1")
col1, col2, col3, col4 = st.columns([2, 1, 2, 1])

with col1:
    descripcion_1 = st.selectbox("IF", [""] + list(opciones_antecedentes.keys()), key="antecedente_1")

with col2:
    es_o_no_1 = st.selectbox("IS/IS NOT", ["IS", "IS NOT"], key="es_o_no_1")

with col3:
    conjunto_1 = ""
    if descripcion_1:
        conjuntos_opciones_1 = opciones_antecedentes[descripcion_1]["fuzzy_sets"]
        conjunto_1 = st.selectbox("", [""] + conjuntos_opciones_1, key="conjunto_1")

with col4:
    operador_1 = st.selectbox("Operator", ["", "AND", "OR"], key="operador_1")

# Antecedente 2
if operador_1:
    st.subheader("Antecedent 2")
    col1, col2, col3, col4 = st.columns([2, 1, 2, 1])

    with col1:
        descripcion_2 = st.selectbox("IF", [""] + list(opciones_antecedentes.keys()), key="antecedente_2")

    with col2:
        es_o_no_2 = st.selectbox("IS/IS NOT", ["IS", "IS NOT"], key="es_o_no_2")

    with col3:
        conjunto_2 = ""
        if descripcion_2:
            conjuntos_opciones_2 = opciones_antecedentes[descripcion_2]["fuzzy_sets"]
            conjunto_2 = st.selectbox("", [""] + conjuntos_opciones_2, key="conjunto_2")

    with col4:
        operador_2 = st.selectbox("Operator", ["", "AND", "OR"], key="operador_2")
else:
    operador_2 = None  # Inicializar operador_2 para evitar errores

# Antecedente 3 (solo visible si hay un operador en el antecedente 2)
if operador_2:
    st.subheader("Antecedent 3")
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        descripcion_3 = st.selectbox("IF", [""] + list(opciones_antecedentes.keys()), key="antecedente_3")

    with col2:
        es_o_no_3 = st.selectbox("IS/IS NOT", ["IS", "IS NOT"], key="es_o_no_3")

    with col3:
        conjunto_3 = ""
        if descripcion_3:
            conjuntos_opciones_3 = opciones_antecedentes[descripcion_3]["fuzzy_sets"]
            conjunto_3 = st.selectbox("", [""] + conjuntos_opciones_3, key="conjunto_3")

# Consecuente
st.subheader("Consequent")

# Desplegable para seleccionar el consecuente
descripcion_consecuente = st.selectbox("THEN", [""] + list(opciones_consecuentes.keys()), key="consecuente")

# Desplegable de conjuntos borrosos para el consecuente seleccionado
if descripcion_consecuente:
    conjuntos_consecuente = opciones_consecuentes[descripcion_consecuente]["fuzzy_sets"]
    conjunto_consecuente = st.selectbox("IS", [""] + conjuntos_consecuente, key="conjunto_consecuente")

# Botón para guardar la regla
if st.button("Save Rule"):
    antecedente_texto = construir_antecedente()
    if descripcion_consecuente and conjunto_consecuente:
        # Obtener el nombre en español para el consecuente y el conjunto en español
        consecuente_es = opciones_consecuentes[descripcion_consecuente]["variable_name_es"]
        index_conjunto_consecuente = opciones_consecuentes[descripcion_consecuente]["fuzzy_sets"].index(conjunto_consecuente)
        conjunto_consecuente_es = opciones_consecuentes[descripcion_consecuente]["fuzzy_sets_es"][index_conjunto_consecuente]
        
        #Preparamos la inserción de la regla a través del endpoint correspondiente
        regla = {
            "antecedent": antecedente_texto,
            "consequent": consecuente_es,
            "consequent_value": conjunto_consecuente_es
        }
        print(regla, type(consecuente_es), type(conjunto_consecuente_es))
        response = requests.post(endponint_reglas, json=regla)
        print(response)
        if response.status_code == 201:
            st.success("Rule saved successfully!")
        else:
            st.error("Error saving the rule.")
    else:
        st.error("Please complete the consequent selection.")
