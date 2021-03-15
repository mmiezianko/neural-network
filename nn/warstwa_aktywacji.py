from nn.warstwy import Warstwa
import numpy as np
from nn.optymalizatory import Optymalizator
from typing import *

class WarstwaAktywacji(Warstwa):
    def __init__(self, f_aktywacji, derr_f_aktywacji):
        self.f_aktywacji = f_aktywacji
        self.derr_f_aktywacji = derr_f_aktywacji

    def forward_prop(self, input: np.ndarray):
        """Obejmuje input funkcją aktywacji"""
        self.X = input
        self.out = self.f_aktywacji(self.X)
        return self.out

    def backward_prop(self,lrn_rate, derr_output_loss, iteracje = None, optymalizator = None):
        """Pochodna z funkcji kosztu po inpucie: ∂E/∂X = ∂E/∂Y * f'(X)"""
        return self.derr_f_aktywacji(self.out) * derr_output_loss
