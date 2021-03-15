import  numpy as np


def podzial(x, y, proc):
    idx_podzialu = int((len(x)*(1-proc)))
    x_train = x[:idx_podzialu]
    x_test = x[idx_podzialu:]
    y_train = y[:idx_podzialu]
    y_test = y[idx_podzialu:]
    return x_train, y_train, x_test, y_test
