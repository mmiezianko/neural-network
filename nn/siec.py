from nn.warstwy import *
import numpy as np
from typing import *
from nn.funkcje_pomocnicze import podzial,accuracy
from nn.optymalizatory import Optymalizator

class Siec:
    def __init__(self):
        self.warstwy: List[Warstwa] = []
        self.f_celu = None
        self.derr_f_celu = None

    def dodaj_warstwe(self, warstwa):
        """Funkcja dodająca warstwę do sieci"""
        self.warstwy.append(warstwa)

    def ust_f_celu(self, f_celu, derr_f_celu):
        """Ustawienie funkcji celu"""
        self.f_celu = f_celu
        self.derr_f_celu = derr_f_celu

    def predykcja(self, input: np.ndarray) -> np.ndarray:
        """
        Funkcja, która implementuje forward propagation na neuronach
        :param input: dane wejściowe
        :return: sygnał neuronu: dane * wagi + bias
        """
        il_obserwacji = len(input)
        wynik = []

        #uruchomienie sieci na wszystkich próbkach i warstwach

        for i in range (il_obserwacji):
            #forward propagation
            x = input[i]

            for warstwa in self.warstwy:
                x = warstwa.forward_prop(x)
            wynik.append(x)

        return np.array(wynik)

        #def update(self, t, w, b, dw, db):

    def trenuj(self, x_train, y_train, iteracje, lrn_rate, proc_walidacyjny = None, batch_size = None):
        historia_trening_loss = []
        historia_walidacja_loss = []
        historia_trening_acc = []
        historia_walidacja_acc = []
        
        walidacja = False
        x_val, y_val = ([], [])
        if proc_walidacyjny is not None:
            walidacja = True
            x_train, y_train, x_val, y_val = podzial(x_train, y_train, proc_walidacyjny)
        il_obserwacji = len(x_train)

        if batch_size is None:
            for i in range(iteracje):
                blad_temp = 0 #zmienna służąca do przechowywania błędu
                historia_trening_loss_sample = []
                historia_walidacja_loss_sample = []
                for j in range(il_obserwacji):

                    # forward propagation
                    x = x_train[j]
                    for warstwa in self.warstwy:
                        #przeprowadza forward propagation na typie danej warstwy, czyli jeśli
                        #wprowadzimy warstwę FC to wykona forward prop dla FC, a jeśli warstwę aktywacji
                        #to wykona forward prop dla warstwy aktywacji
                        x = warstwa.forward_prop(x)

                    #obliczamy loss aby później można go było wyświetlić
                    blad_temp += self.f_celu(y_train[j], x)
                    historia_trening_loss_sample.append(blad_temp)
                    #backward propagation
                    blad = self.derr_f_celu(y_train[j], x)
                    for warstwa in reversed(self.warstwy):
                        # reversed -> bierzemy warstwy "od końca"
                        # i iterujemy po kolejnych obserwacjach aktualizując wagi
                        blad = warstwa.backward_prop(lrn_rate, blad, iteracje=i)

                    if walidacja:
                        pred_walid = self.predykcja(x_val)
                        blad_walid = self.f_celu(y_val, pred_walid)
                        historia_walidacja_loss_sample.append(blad_walid)
                historia_trening_loss.append(np.average(historia_trening_loss_sample))
                historia_walidacja_loss.append(np.average(historia_walidacja_loss_sample))
                if walidacja:
                    historia_walidacja_acc.append(accuracy(y_val,self.predykcja(x_val)))
                historia_trening_acc.append(accuracy(y_train, self.predykcja(x_train)))
        else:
            X = x_train
            np.random.seed(42)
            il_batchy = il_obserwacji//batch_size #ilosc batchy musi byc liczba calkowita
            reszta = il_obserwacji-il_batchy*batch_size
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[-1])
            y_train = y_train.reshape(y_train.shape[0], 1)
            dane = np.concatenate((x_train, y_train), axis=1)
            np.random.shuffle(dane)
            x_train = dane[:, :-1]
            y_train = dane[:,-1]
            for i in range(iteracje):
                blad_temp = 0  # zmienna służąca do przechowywania błędu
                historia_trening_loss_sample = []
                historia_walidacja_loss_sample = []

                for j in range(il_batchy):
                    #dzielimy nasze dane na batche czyli podzbiory wg algorytmu, ze do kazdego rzędu danych dobieramy ustaloną ilość obserwacji (argument batch_size)

                    if j < il_batchy:
                        x = x_train[j*batch_size:(j+1)*batch_size, :]
                        y = np.expand_dims(y_train[j*batch_size:(j+1)*batch_size],axis = -1)
                    else:
                        x_train = x_train.reshape(x_train.shape[0],x_train.shape[-1])
                        y_train = y_train.reshape(y_train.shape[0],1)
                        dane = np.concatenate((x_train, y_train), axis=1)
                        np.random.shuffle(dane)
                        dane_uz = np.random.choice([x for x in range(len(dane))], batch_size - reszta)
                        x = np.concatenate((x_train[j*batch_size:, :], np.delete(dane[dane_uz, :],  -1, axis=1)), axis=0)
                        y = np.concatenate((y_train[j*batch_size:], dane[dane_uz, -1]), axis=0)


                    #forward propagation

                    x =x.reshape(batch_size, x.shape[-1])
                    #x ma wymiar batch x feature

                    for warstwa in self.warstwy:
                        # przeprowadza forward propagation na typie danej warstwy, czyli jeśli
                        # wprowadzimy warstwę FC to wykona forward prop dla FC, a jeśli warstwę aktywacji
                        # to wykona forward prop dla warstwy aktywacji
                        x = warstwa.forward_prop(x)

                    # obliczamy loss aby później można go było wyświetlić
                    blad_temp += self.f_celu(y, x)

                    historia_trening_loss_sample.append(blad_temp)
                    # backward propagation
                    blad = self.derr_f_celu(y, x)



                    for warstwa in reversed(self.warstwy):
                        # reversed -> bierzemy warstwy "od końca"
                        # i iterujemy po kolejnych obserwacjach aktualizując wagi

                        blad = warstwa.backward_prop(lrn_rate, blad, iteracje=i)

                    if walidacja:
                        pred_walid = self.predykcja(x_val)

                        blad_walid = self.f_celu(y_val, np.array(pred_walid).squeeze(axis=-1))
                        historia_walidacja_loss_sample.append(blad_walid)
                        
                if i==iteracje-1:
                    print(blad_temp)

                historia_trening_loss.append(np.average(historia_trening_loss_sample))
                historia_walidacja_loss.append(np.average(historia_walidacja_loss_sample))
                if walidacja:
                    historia_walidacja_acc.append(accuracy(y_val, self.predykcja(x_val)))
                historia_trening_acc.append(accuracy(y_train, self.predykcja(x_train)))




        return {'blad_walidacji': historia_walidacja_loss, 'blad_trening': historia_trening_loss,
                'dokładnosc_walidacji': historia_walidacja_acc, 'dokładnosc_treningowa': historia_trening_acc }

