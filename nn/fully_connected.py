from nn.warstwy import Warstwa
import numpy as np
from typing import *
from nn.optymalizatory import Optymalizator

class WarstwaFC(Warstwa):
    """Implementacja warstwy w pełni połączonej """

    def __init__(self, n_neurons_in: int, n_neurons_out: int, optymalizator: Optymalizator = None):
        #n_neurons_inp to liczba neuronów na start
        #n_neurons_out to liczba neuronów na output
        self.optymalizator = optymalizator
        self.bias = np.random.rand(1, n_neurons_out) - 0.5
        """ Macierz liczb pseudolosowych o wymiarach 1 x n_neurons_out (liczba neuronów do outputu)"""
        self.wagi = np.random.rand(n_neurons_in, n_neurons_out) - 0.5
        """ Macierz liczb pseudolosowych o wymiarach n_neurons_in (liczba neuronów na start) x n_neurons_out (liczba neuronów do outputu)"""

    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        """
        Iloczyn macierzowy inputu oraz wag powiększony o bias. Zwraca output
        :param input: dane wejściowe typu ndarray
        :return: sygnał wyjściowy
        """
        self.X = input
        self.out = self.X @ self.wagi + self.bias
        return self.out

    def backward_prop(self, lrn_rate, derr_output_loss, iteracje=None, **kwargs):
        """ Obliczamy gradient, dzięki któremu skorygowane zostaną wagi. W tym celu wyznaczamy pochodną funkcji kosztu
        z uwzględnieniem parametrów(bias, wagi) oraz pochodną po funkcji kosztu z uwzględnieniem inputów.
        Do obliczania pochodnych zastosowana została reguła łańcuchowa  """

        # Liczymy dE/dW, dE/dB dla zadanego derr_out_loss=dE/dY :
        # dot = element wise multiplication (w odróżnieniu od iloczynu skalarnego nie sumuje wyników mnożenia)
        derr_X_loss = np.dot(derr_output_loss, self.wagi.T)
        #dajemy .T ponieważ jest to atrybut npdarray i działa szybciej niż transpose które działa za pomocą pętli
        """Pochodna z funkcji kosztu po inpucie: ∂E/∂X = ∂E/∂Y * transponowane wagi  """

        gradient = np.dot(self.X.T, derr_output_loss)
        """Pochodna z funkcji kosztu po parametrze wagi: ∂E/∂W = transponowany input * ∂E/∂Y  """
        # Pochodna po bias to ∂E/∂Y czyli derr_output_loss

        if self.optymalizator is None:

            self.wagi -= lrn_rate * gradient
            """Aktualizacja wag o gradient"""
            self.bias -= (np.full((1,derr_output_loss.shape[0]),lrn_rate) @ derr_output_loss)/derr_output_loss.shape[0]
            '''
            Aktualizacja biasu o jego pochodną i learning rate.
            '''

        else:
            self.wagi, self.bias = self.optymalizator.update(iteracje, self.wagi, self.bias, gradient, derr_output_loss)
            # ∂E/∂W to gradient
            # derr_output_loss

        return derr_X_loss



