import streamlit as st
from neuron import Neuron


# Configuramos la página para visualización tipo wide, cargamos una imagen y título
st.set_page_config(layout="wide")
col3, col4, col5 = st.columns(3)  # Podemos cada input uno al lado del otro utilizando 2 columnas
with col3:
    st.image('images/neurona3.png', width=350)
with col4:
    st.image('images/neurona3.png', width=350)
with col5:
    st.image('images/neurona3.png', width=350)

st.title('Simulador de Neurona')

st.header("Neurona generica con función de activación (perceptrón simple)")

# Barra de selección de número de entradas y pesos posibles
n_w_x = st.slider('Elige el número de entradas/pesos que tendrá la neurona', 1, 10, 1)

st.subheader("Pesos", divider='rainbow')
pesos = []   # Creamos una lista vacia para ir almacenando pesos
# Con un bucle vamos creando columnas segun el numero de pesos y entradas seleccionadas en el slider
for col in st.columns(n_w_x):
    with col:
        # En cada columna vamos metiendo los valores de los pesos y 
        w = st.number_input(f'w$_{len(pesos)}$', key=f'w{len(pesos)}')  # Segun la longitud del input crea una clave unica que identifica a cada campo
        pesos.append(w) # Los pesos ontenidos en los inputs se van almacenando en la lista vacía
st.text(f'x = {pesos}') # Los mostramos debajo del input en forma de texto


# Hacemos exactamente lo mismo para la sección entradas
st.subheader('Entradas', divider='rainbow')
entradas = []  

for col in st.columns(n_w_x):
    with col:
        x = st.number_input(f'x$_{len(entradas)}$', key=f'x{len(entradas)}')
        entradas.append(x)
st.text(f'x = {entradas}')


# Sección del sesgo y la función de activación
col1, col2 = st.columns(2)  # Podemos cada input uno al lado del otro utilizando 2 columnas
with col1:
    st.subheader('Valor del Sesgo', divider='rainbow')
    b = st.number_input("Introduzca el valor del sesgo", value=0.00, step=0.01)

with col2:
    st.subheader('Funcion de activación', divider='rainbow')
    activation_f = st.selectbox(
    "Elige la función de activación",
    ("Sigmoid", "ReLU", "Tanh"),
    index=0,
    placeholder="Elige la función de activación",
    disabled=False, 
    label_visibility="hidden"
)


st.subheader('Valor de salida', divider='rainbow')
if st.button('Calcular la salida'):
    st.divider()
    # Creamos una instancia de la neurona llamando a la clase "Neuron"
    neurona = Neuron(pesos, b, activation_f)
    output = neurona.run(entradas)  # Calculamos la salida
    st.text(f"La salida de la neurona es {output}")

st.divider()

col6, col7, col8 = st.columns(3)  # Podemos cada input uno al lado del otro utilizando 2 columnas
with col6:
    st.image('images/neurona3.png', width=350)
with col7:
    st.image('images/neurona3.png', width=350)
with col8:
    st.image('images/neurona3.png', width=350)

st.markdown("© Jesús Cánovas Barqueros - CPIFP Alan Turing")
