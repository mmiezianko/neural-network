from abc import ABC, abstractmethod


class Warstwa(ABC):
    def __init__(self):
        self.X = None #input = X
        self.out = None #output = Y

    @abstractmethod
    def forward_prop(self, X):
        """ Przebieg informacji od warstwy wejściowej do warstwy wyjściowej"""
        pass

    @abstractmethod
    def backward_prop(self,lrn_rate, derr_output_loss, iteracje = None, optymalizator = None ):
        """Oblicza dE / dX dla danego dE / dY (gradienty) oraz aktualizuje wagi """
        pass



