import  numpy as np


def podzial(x, y, proc):
    idx_podzialu = int((len(x)*(1-proc)))
    x_train = x[:idx_podzialu]
    x_test = x[idx_podzialu:]
    y_train = y[:idx_podzialu]
    y_test = y[idx_podzialu:]
    return x_train, y_train, x_test, y_test

def train_test_split(data,test_split=0.2, seed=42):
    idx_podzialu = int((len(data) * (1 - test_split)))
    np.random.seed(seed)
    np.random.shuffle(data)
    x = data[:,:-1]
    y = data[:,-1]
    x_train = x[:idx_podzialu]
    x_test = x[idx_podzialu:]
    y_train = y[:idx_podzialu]
    y_test = y[idx_podzialu:]
    return x_train, y_train, x_test, y_test

def accuracy(y_true,y_pred):
    is_eq = np.array(np.round(y_pred.flatten()) == y_true.flatten())
    #print(is_eq.shape)
    return np.sum(is_eq)/len(y_true)


