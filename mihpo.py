#-*- coding: utf-8 -*-
# Written by Hyunki Seong.
# Email : hynkis@kaist.ac.kr

import sys
import numpy as np
import ujson as json
from math import log, ceil

class MIHO:
    def __init__(self, model, R=1e+5, eta=4):
        """
        Model Identification via Hyperparameter Optimization (MIHO)
            - R: the maximum amount of resource.
            - eta: a value that determines the proportion of the discarded configurations.
        """
        self.model = model
        
        self.max_iter = R
        self.eta = eta
        self.s_max = int(log(R) / log(eta))
        self.B = (self.s_max + 1) * R  
        
        self.best_obj = np.inf
        self.best_config = None
        self.history = []

        print("s_max: %.3f, B: %.3f" %(self.s_max, self.B))

    def run(self):
        for s in reversed(range(self.s_max + 1)):
    
            """""""""""""""""""""""""""""""""
            -----     Initialize n     -----
            """""""""""""""""""""""""""""""""
            # initial number of configs
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s)) 
            
            """""""""""""""""""""""""""""""""
            -----     Initialize r     -----
            """""""""""""""""""""""""""""""""
            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s) 
            
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            -----     Initialize config (get model param config)     -----
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            # initial configs
            configs = [self.model.rand_config() for _ in range(n)] 
            
            for i in range(s + 1):
                # number of iterations for these configs
                r_i = r * self.eta ** i
                print(f"-- {len(configs):d} configs @ {int(round(r_i)):d} iterations --")

                results = []
                num_configs = len(configs)

                for j, config in enumerate(configs):
                    """""""""""""""""""""""""""""""""""
                    -----    eval with mutation   -----
                    """""""""""""""""""""""""""""""""""
                    res = self.model.eval_config(config=config, iters=r_i, same_with_best_config=(config == self.best_config))
                    results.append(res)
                    
                    self.best_obj = min(res['obj'], self.best_obj)
                    if self.best_obj == res['obj']:
                        self.best_config = res['config']
                    print(f"Current: {float(res['obj']):.3f} | Best: {self.best_obj:.3f}")
                    print("Current s : {}, i/i_max : {}/{}, i_config/num_config : {}/{}, best config : {}, obj : {}".format(s, i, s+1, j,num_configs, self.best_config, self.best_obj))
                    sys.stdout.flush()
                
                self.history += results

                """""""""""""""""""""""""""""""""
                -----  select top k config  -----
                """""""""""""""""""""""""""""""""
                # - Sort by objective value
                results = sorted(results, key=lambda x: x['obj'])
                
                # - Drop models that have already converged
                results = list(filter(lambda x: not x.get('converged', False), results))
                
                # - Determine how many configs to keep
                n_keep = int(n * self.eta ** (-i - 1))
                configs = [result['config'] for result in results[:n_keep]]

