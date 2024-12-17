import streamlit as st
import requests

st.set_page_config(layout="wide", page_title="Rules Creation")


import requests

BASE_URL = "http://api:8000"

# API Endpoints
endpoint_antecedentes = f"{BASE_URL}/antecedents"
endpoint_consecuentes = f"{BASE_URL}/consequents"
endpoint_reglas = f"{BASE_URL}/create_rule/"

# Obtaining data from endpoints
@st.cache_data
def obtener_datos(endpoint):
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error al obtener datos del endpoint")
        return []


datos_antecedentes = obtener_datos(endpoint_antecedentes)
datos_consecuentes = obtener_datos(endpoint_consecuentes)

# Create dictionaries to map ‘variable_description’ to the required data in alphabetical order.
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


st.title("Creation of Transversal Skills Assessment Rules")

# Function to construct the antecedent
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
                # Obtain values in English for the variable and fuzzy set.
                variable_es = opciones_antecedentes[variable]["variable_name_es"]
                index_conjunto = opciones_antecedentes[variable]["fuzzy_sets"].index(conjunto)
                conjunto_es = opciones_antecedentes[variable]["fuzzy_sets_es"][index_conjunto]
                
                
                condicion_texto = f"{'NOT ' if es_o_no == 'IS NOT' else ''}{variable_es}[{conjunto_es}]"
                antecedente += condicion_texto
                if operador:  # Add operator (AND/OR) if it is not the last one
                    antecedente += f" {operador} "
                    
    return 'IF '+ antecedente

# Antecedent 1
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

# Antecedent 2
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
    operador_2 = None  # Initialise operador_2 to avoid errors

# Antecedent 3 (only visible if there is an operator in antecedent 2)
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

# Consequent
st.subheader("Consequent")

# Drop-down to select the consequent
descripcion_consecuente = st.selectbox("THEN", [""] + list(opciones_consecuentes.keys()), key="consecuente")

# Drop-down of fuzzy sets for the selected consequent
if descripcion_consecuente:
    conjuntos_consecuente = opciones_consecuentes[descripcion_consecuente]["fuzzy_sets"]
    conjunto_consecuente = st.selectbox("IS", [""] + conjuntos_consecuente, key="conjunto_consecuente")


if st.button("Save Rule"):
    antecedente_texto = construir_antecedente()
    if descripcion_consecuente and conjunto_consecuente:
        # Get the name in Spanish for the consequent and the whole in Spanish
        consecuente_es = opciones_consecuentes[descripcion_consecuente]["variable_name_es"]
        index_conjunto_consecuente = opciones_consecuentes[descripcion_consecuente]["fuzzy_sets"].index(conjunto_consecuente)
        conjunto_consecuente_es = opciones_consecuentes[descripcion_consecuente]["fuzzy_sets_es"][index_conjunto_consecuente]
        
        # Preparing the insertion of the rule through the corresponding endpoint.
        regla = {
            "antecedent": antecedente_texto,
            "consequent": consecuente_es,
            "consequent_value": conjunto_consecuente_es
        }
        print(regla, type(consecuente_es), type(conjunto_consecuente_es))
        response = requests.post(endpoint_reglas, json=regla)
        print(response)
        if response.status_code == 201:
            st.success("Rule saved successfully!")
        else:
            st.error("Error saving the rule.")
    else:
        st.error("Please complete the consequent selection.")
