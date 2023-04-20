#-*- coding: utf-8 -*-
# Written by Hyunki Seong.
# Email : hynkis@kaist.ac.kr
import sys
import glob
import numpy as np
import pandas as pd
import csv
import time
from mihpo import MIHPO
from mutation_model import ExponentialGaussianDistribution

sys.path.insert(1, './model')
from tire_model import lateral_tire_model

class TireModelID:
    """
    - Lateral Tire Model Identification (Pacejka tire model)
        F_y = Sy + D * np.sin(C * np.arctan2(B * np.deg2rad(slip_angle + Sx)))
        - D : pacejka tire model parameter
        - C : pacejka tire model parameter
        - B : pacejka tire model parameter
        - Sx : offset parameter (x-axis)
        - Sy : offset parameter (y-axis)

    - Parameters to optimize
        - Parameters of the Pacejka tire model: [B, C, D, Sx, Sy]

    - Constant parameters:
        - self.m : tire mass
    """

    def __init__(self, mutation_model, data_buffer):
        # Noise model for perturbation
        self.mutation_model = mutation_model

        # Static params
        self.m = 150 # tire mass

        # Middle value of hyper param with trivial heuristic
        self.B_mean  = 100
        self.C_mean  = 100
        self.D_mean  = self.m*9.81*1.5
        self.Sx_mean = 0
        self.Sy_mean = 0

        # Min Max Range of hyper param with trivial heuristic
        self.B_min  = 0
        self.C_min  = 0
        self.D_min  = self.D_mean * 0.1
        self.Sx_min = self.Sx_mean - 0.1 # [rad]
        self.Sy_min = self.Sy_mean - 0.1 * self.D_mean # [N]

        self.B_max  = self.B_mean * 2.0
        self.C_max  = self.C_mean * 2.0
        self.D_max  = self.D_mean * 2.0
        self.Sx_max = self.Sx_mean + 0.1 # [rad]
        self.Sy_max = self.Sy_mean + 0.1 * self.D_mean # [N]

        # Params for sampling
        self.n_sample_B  = 10000
        self.n_sample_C  = 10000
        self.n_sample_D  = 10000
        self.n_sample_Sx = 10000
        self.n_sample_Sy = 10000

        # Params for noise
        self.sigma_init_B  = 5.0
        self.sigma_init_C  = 5.0
        self.sigma_init_D  = 100.0
        self.sigma_init_Sx = 0.1
        self.sigma_init_Sy = 0.1

        self.sigma_end_B  = 0.0
        self.sigma_end_C  = 0.0
        self.sigma_end_D  = 0.0
        self.sigma_end_Sx = 0.0
        self.sigma_end_Sy = 0.0

        # Data buffer. list of (slip_angle, F_y_true). 
        self.data_buffer = data_buffer # (N,2,dim). 

    def rand_config(self):
        """
        Returns random configuration

        """
        B = np.random.choice(np.linspace(self.B_min, self.B_max, self.n_sample_B)) + self.sigma_init_B * np.random.normal(0, 1)
        C = np.random.choice(np.linspace(self.C_min, self.C_max, self.n_sample_C)) + self.sigma_init_C * np.random.normal(0, 1)
        D = np.random.choice(np.linspace(self.D_min, self.D_max, self.n_sample_D)) + self.sigma_init_D * np.random.normal(0, 1)
        Sx = np.random.choice(np.linspace(self.Sx_min, self.Sx_max, self.n_sample_Sx)) + self.sigma_init_Sx * np.random.normal(0, 1)
        Sy = np.random.choice(np.linspace(self.Sy_min, self.Sy_max, self.n_sample_Sy)) + self.sigma_init_Sy * np.random.normal(0, 1)
        
        # Minimum condition
        B = np.maximum(self.B_min, B)
        C = np.maximum(self.C_min, C)
        D = np.maximum(self.D_min, D)
        Sx = np.maximum(self.Sx_min, Sx)
        Sy = np.maximum(self.Sy_min, Sy)

        return [B, C, D, Sx, Sy] # should be list for json
        
    def eval_config(self, config, iters, same_with_best_config):
        """
        Evaluates the model with a given parameter configuration using a given dataset.
            - config : [B, C, D, Sx, Sy]
        """
        tic = time.time()

        # - Get sampled params
        params = config # [B, C, D, Sx, Sy]
        
        # - Get data
        slip_angle_buffer = self.data_buffer[:,0] # (N,)
        F_y_true_buffer   = self.data_buffer[:,1] # (N,)
        
        # - Compute model output
        F_y_buffer = lateral_tire_model(slip_angle_buffer, params)
        
        # - Initial evaluation
        loss = F_y_buffer - F_y_true_buffer        
        loss = np.array(loss)
        loss = np.mean(loss**2) # MSE

        # - Save config before eval_with_mutation
        best_config = config
        best_loss = loss

        # - Evaluation with the Gaussian mutation (Gaussian distribution exponentially annealed sigma) 
        config = np.array(config)
        mu = np.zeros_like(config)
        sigma_init = np.array([self.sigma_init_B, self.sigma_init_C, self.sigma_init_D, self.sigma_init_Sx, self.sigma_init_Sy])
        sigma_end  = np.array([self.sigma_end_B,  self.sigma_end_C,  self.sigma_end_D, self.sigma_end_Sx, self.sigma_end_Sy])
        
        # - noise for mutation
        noise = self.mutation_model(mu=mu, sigma_init=sigma_init, sigma_end=sigma_end, max_epoch=iters, halving_ratio=0.25)

        # - Do learning iteration (during resource allocation, 'r')
        for n in range(int(iters)):
            # - Get new params
            new_config = best_config + noise(n)
            # - minimum condition
            new_config = np.maximum(new_config,
                                    [self.B_min, self.C_min, self.D_min, self.Sx_min, self.Sy_min],
                                    )

            new_params = new_config # [B, C, D, Sx, Sy]

            # - Calculate new validation loss with current configuration
            F_y_buffer = lateral_tire_model(slip_angle_buffer, new_params)
            new_loss = F_y_buffer - F_y_true_buffer
            new_loss = np.array(new_loss)
            new_loss = np.mean(new_loss**2) # MSE

            # - Check whether global best config is optimized or not
            if same_with_best_config:
                print("Same with global best config!!")
                print("Original config :", config, best_loss)
                print("New config      :", new_config, new_loss)

                if new_loss < best_loss:
                    print("global best config is changed!!")
                    print("[before]: ", best_config, best_loss)
                    print("[after] config: ", new_config, new_loss)

            # - Optimize parameter with current configuration
            if new_loss < best_loss:
                best_config = new_config
                best_loss   = new_loss

        toc = time.time()
        print("eval process time :", round(toc-tic, 3))

        # - Return optimized parameter w.r.t. current configuration
        return {
            "config" : list(best_config),  # should be list for json
            "obj" : best_loss
        }

# Run main process
if __name__ == "__main__":
    LOAD_CSV_PATH = './csv_files/lateral_model/tire_data_SA_FyA_FyB.csv'
    SAVE_RESULT_PATH = './results/result_tire_param_SA_FyA_FyB.csv'

    f = open(SAVE_RESULT_PATH, 'w')
    thewriter = csv.writer(f)
    thewriter.writerow(["Tire", "B", "C", "D", "Sx", "Sy", "Loss"])
    
    # Read csv file
    print(LOAD_CSV_PATH)
    dataset = pd.read_csv(LOAD_CSV_PATH)
    dataset_length = dataset.shape[1] - 1 # except slip_angle
    # - load data for input state and label (tire_data_SA_FyA_FyB.csv)
    # -- dataset[:,0] : slip_angle
    # -- dataset[:,1] : Fy_A
    # -- dataset[:,2] : Fy_B
    for i in range(dataset_length):
        slip_angle = dataset.iloc[:,0].to_numpy()
        F_y_true   = dataset.iloc[:,i+1].to_numpy()

        # -- data buffer (N,2)
        data_buffer = np.array([slip_angle, F_y_true]).T
        print("Loading data has done :", data_buffer.shape)
        
        model = TireModelID(mutation_model=ExponentialGaussianDistribution, data_buffer=data_buffer)
        optimizer = MIHPO(model, R=1e+5, eta=4)
        optimizer.run() # R=1e+5, eta=5

        # - Save learned model parameters
        tire_name = "Fy_A"
        if i+1 == 1:
            tire_name = "Fy_A"
        elif i+1 == 2:
            tire_name = "Fy_B"
        
        thewriter.writerow([
                    tire_name,
                    optimizer.best_config[0], # B
                    optimizer.best_config[1], # C
                    optimizer.best_config[2], # D
                    optimizer.best_config[3], # Sx
                    optimizer.best_config[4], # Sy
                    optimizer.best_obj,
                    ])

