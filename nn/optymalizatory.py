from abc import abstractmethod

import numpy as np


class Optymalizator():

    @abstractmethod
    def update(self, t, w, b, dw, db):
        pass


class AdamOptymalizator(Optymalizator):
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w, b, dw, db):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        # *** biases *** #
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        # *** biases *** #
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)

        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**(t+1))
        m_db_corr = self.m_db/(1-self.beta1**(t+1))
        v_dw_corr = self.v_dw/(1-self.beta2**(t+1))
        v_db_corr = self.v_db/(1-self.beta2**(t+1))
        print(v_db_corr)
        ## update weights and biases
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        b = b - self.eta*(m_db_corr/(np.sqrt(np.abs(v_db_corr))+self.epsilon))
        return w, b
