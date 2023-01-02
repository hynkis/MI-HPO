#-*- coding: utf-8 -*-
# Written by Hyunki Seong.
# Email : hynkis@kaist.ac.kr

import numpy as np
import matplotlib.pyplot as plt

class ExponentialGaussianDistribution():
    def __init__(self, mu, sigma_init, sigma_end, max_epoch=1000, halving_ratio=0.25):
        """
        Gaussian Distribution with exponentially annealed sigma.
            sigma = sigma_init * exp(-a * n)
            sigma_init : initial sigma
            sigma_end  : sigma to be converged
            n  : epoch num
            a  : for half point of the noise during epochs.
               : a = ln(2)/(n / halving_ratio)
               : ex) if halving_ratio := 1/4, 0.5 * sigma_init == sigma_init * e(-a * n/4)
        """
        self.mu    = mu
        self.sigma_init = sigma_init
        self.sigma_end  = sigma_end
        self.max_epoch  = max_epoch
        self.halving_ratio = halving_ratio
        self.gamma = np.log(2) / (max_epoch * halving_ratio) # for half point of the noise during epochs. 

    def __call__(self, epoch_num):
        sigma = self.sigma_init * np.exp(-self.gamma * epoch_num)
        return self.mu + sigma * np.random.normal(size=self.mu.shape)

