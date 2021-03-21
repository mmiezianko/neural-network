import numpy as np
from nn.funkcje_celu import MSE, BinaryCrossEntropy
from nn.siec import *
from nn.fully_connected import WarstwaFC
from nn.funkcje_aktywacji import *
from nn.warstwa_aktywacji import WarstwaAktywacji
from nn.optymalizatory import AdamOptymalizator
import sklearn.datasets as sd
# training data
# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
x_train, y_train= sd.make_classification(n_samples=100)
x_train = np.expand_dims(x_train, axis=1)
y_train = np.expand_dims(y_train, axis=1)
print(x_train.shape)
# x_train = np.reshape(x_train, (1,100, 20))
# y_train = np.reshape(y_train, (1,100, 20))
# network
net = Siec()
net.dodaj_warstwe(WarstwaFC(20, 64, optymalizator=AdamOptymalizator()))
net.dodaj_warstwe(WarstwaAktywacji(sigmoid, derr_sigmoid))
net.dodaj_warstwe(WarstwaFC(64, 64, optymalizator=AdamOptymalizator()))
net.dodaj_warstwe(WarstwaAktywacji(sigmoid, derr_sigmoid))
net.dodaj_warstwe(WarstwaFC(64, 1, optymalizator=AdamOptymalizator()))
net.dodaj_warstwe(WarstwaAktywacji(sigmoid, derr_sigmoid))

m = MSE()
bin_cross_entropy = BinaryCrossEntropy()

# train
net.ust_f_celu(bin_cross_entropy.funk, bin_cross_entropy.derr)
historia = net.trenuj(x_train, y_train, iteracje=50, lrn_rate=0.01, proc_walidacyjny=0.2, batch_size= 32)

# test
out = net.predykcja(x_train)
import matplotlib.pyplot as plt
fix1,ax1 = plt.subplots()
ax1.plot(historia['blad_trening'], scaley = True)
ax1.plot(historia['blad_walidacji'], scaley = True)
ax1.legend(['Błąd teningowy', 'Błąd walidacyjny'])
#plt.ylim([0,1.5])
fix1.show()
fix,ax = plt.subplots()
ax.plot(historia['dokładnosc_walidacji'], scaley = True)
ax.legend(['Dokładność treningowa', 'Dokładność walidacyjna'])
#plt.ylim([0,1.5])
fix.show()

