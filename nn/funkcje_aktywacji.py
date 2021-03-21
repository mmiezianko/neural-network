import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derr_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def tgh(x):
    """Tangens hyperboliczny - funkcja podobna do sigmoidy. Często
    używany do klasyfikacji binarnej (dwie klasy)"""
    return np.tanh(x);

def derr_tgh(x):
    """Pochodna tangensa hiperbolicznego"""
    return 1 - np.tanh(x) ** 2;

def relu(X):
   return np.maximum(0,X)

def derr_relu(x):
  shape = x.shape
  for i in x.flatten() :
    i = 0 if i < 0.0 else 1

  return  x.reshape(shape)

