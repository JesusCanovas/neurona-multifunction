# Neurona Multifunción

Este proyecto implementa una neurona con diferentes configuraciones con hasta 10 entradas, 10 pesos, 1 sesgo y tres funciones de activación diferentes: ReLU, Sigmoid y Tanh.

<img width="250" height="300" src="images/neurona3.png">


## Contenido

1. [Descripción](#descripción)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Requisitos](#requisitos)
4. [Instrucciones de Uso](#instrucciones-de-uso)
5. [Agradecimientos](#agradecimientos)
   

## Descripción

La aplicación "Neurona Multifunción" proporciona ejemplos interactivos de neuronas con diferentes configuraciones. Puedes visualizar y experimentar con neuronas de 1-10 entradas y pesos, así como sesgo y tres funciones distintas de activación.
Podemos ver una captura de la pariencia de la aplicación y explicar brevemente su funcionamiento:
- Tenemos un "slide bar" para elegir el número de entradas/pesos que tendrá la neurona.
- A continuación en los "inputs box" podemos configurar el valor de los pesos y las entradas.

<img src="images/neurona4.png">

- También podremos ajustar el sesgo.
- Finalmente podemos elegir la función de activación deseada mediante un "selectbox".


## Estructura del Proyecto

- `app.py`: Contiene el código principal de la aplicación con las implementaciones de las neuronas.
- `neuron.py`: Contiene la clase "Neuron" de la que se hace uso en la aplicación y contiene el codigo fuente principal del funcionamiento de la neurona.
- `images/`: Carpeta que contiene imágenes utilizadas en la aplicación.
- `README.md`: Este archivo.


## Requisitos

Asegúrate de tener instalados los siguientes paquetes antes de ejecutar la aplicación:

- `pip install streamlit numpy`
- `pip install streamlit pandas`


## Instrucciones de Uso

1. Clona el repositorio:

- `git clone https://github.com/tuusuario/turepositorio.git`
- `cd turepositorio`

2. Instala los requisitos:
- `pip install -r requirements.txt`

3. Ejecuta la aplicación:
- `streamlit run app.py`


## Agradecimientos

Gracias por utilizar la aplicación "Neurona Multifunción". Si tienes alguna pregunta, puedes contactar al autor:


Jesús Cánovas Barqueros
Email: jesuscanovas3w@gmail.com
