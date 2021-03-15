from abc import abstractmethod
import numpy as np


def algebraiczna_jedynka(size):
    return np.ones((size,  1))

class FunkcjaCelu:
    @abstractmethod
    def __call__(self, y_true, y_pred, *args, **kwargs):
        pass


class MSE(FunkcjaCelu):
    def __init__(self):
        self.derr = lambda y_true,y_pred: 2*(np.sum(y_pred-y_true))/y_true.size
        self.funk = lambda y_true,y_pred: np.mean(np.sum(y_true-y_pred)**2)

    def __call__(self, y_true, y_pred, *args, **kwargs):
        return self.funk(y_true, y_pred)

class BinaryCrossEntropy(FunkcjaCelu):
    def __init__(self):
        self.funk = lambda y_true, y_pred: np.sum(-y_true * np.log(y_pred) - (algebraiczna_jedynka(y_true.shape[0])-y_true)*np.log(algebraiczna_jedynka(y_true.shape[0])-y_pred))/y_true.shape[0]
        self.derr = lambda y_true, y_pred: ((-y_true/y_pred)+(algebraiczna_jedynka(y_true.shape[0])-y_true)/(algebraiczna_jedynka(y_true.shape[0])-y_pred))/y_true.shape[0]

    def __call__(self, y_true, y_pred, *args, **kwargs):
        return self.funk(y_true, y_pred)





