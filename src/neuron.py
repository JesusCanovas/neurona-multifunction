
import pandas as pd
import numpy as np
import math

class Neuron:

    def __init__(self, weights, bias, func):
        """Constructor de la clase Neuron.

        Parámetros:
        - weights: Lista de pesos para las entradas de la neurona.
        - bias: Sesgo de la neurona.
        - func: String que representa la función de activación ("relu", "sigmoid" o "tanh").
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = func  # Almacenamos el nombre de la función

    @staticmethod
    def __relu_activation(x):
        """Función de activación ReLU."""
        return max(0, x)

    @staticmethod
    def __sigmoid_activation(x):
        """Función de activación Sigmoid."""
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def __tanh_activation(x):
        """Función de activación Tangente Hiperbólica."""
        return math.tanh(x)

    def run(self, input_data):
        """Realiza la operación de propagación hacia adelante de la neurona.

        Parámetros:
        - input_data: Lista de datos de entrada.

        Retorna:
        - Salida de la neurona después de aplicar la función de activación.
        """
        weighted_sum = sum(w * x for w, x in zip(self.weights, input_data))
        # Llamamos a la función de activación por su nombre almacenado
        output = getattr(Neuron, f'_{self.__class__.__name__}__{self.activation_function.lower()}_activation')(weighted_sum + self.bias) #Name mangling
        #output = getattr(self, f'_{self.__class__.__name__}__{self.activation_function.lower()}_activation')(weighted_sum + self.bias)
        return output