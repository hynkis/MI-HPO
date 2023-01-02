#-*- coding: utf-8 -*-
# Written by Hyunki Seong.
# Email : hynkis@kaist.ac.kr

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from model.tire_model import lateral_tire_model

"""
Preprocess with learned model
"""
path_dir = './csv_files/lateral_model'
SAVE_TOTAL_DATA = False

LOAD_CSV_PATH = './csv_files/lateral_model/tire_data_SA_FyA_FyB.csv'
dataset = pd.read_csv(LOAD_CSV_PATH)

# tire lateral
slip_angle = dataset.iloc[:,0].to_numpy()
tire_Fy_A = dataset.iloc[:,1].to_numpy()/1000
tire_Fy_B = dataset.iloc[:,2].to_numpy()/1000

"""
Learned Front lateral tire
"""

# found params
# Fy_A,17.197514461629723,1.6717081677023067,2651.105946201609,0.007627728874956934,134.73955296770842,153660.70195698805
# Fy_B,17.472250073164812,1.7576987654036749,3451.8729290576975,0.010287248963999114,194.0794300197951,151085.55279822915
params_A = [17.197, 1.6717, 2651.105, 0.00762, 134.739]  # [B, C, D, Sx, Sy]
params_B = [17.472, 1.7576, 3451.872, 0.01028, 194.079]  # [B, C, D, Sx, Sy]

slip_angle_list = np.linspace(-np.deg2rad(15), np.deg2rad(15), 10000)
Fy_A_list = lateral_tire_model(slip_angle_list, params_A)/1000
Fy_B_list = lateral_tire_model(slip_angle_list, params_B)/1000

"""
Plot data with learned model
"""

# plt.subplot(121)
fig, (ax1) = plt.subplots(1,1)
ax1.axvline(x=0, color='k')
ax1.axhline(y=0, color='k')

ax1.plot(slip_angle, tire_Fy_A, 'o', color='cornflowerblue', alpha=0.2, label="Data (A)")
ax1.plot(slip_angle, tire_Fy_B, 'o', color='lightcoral', alpha=0.2, label="Data (B)")
ax1.plot(slip_angle_list, Fy_A_list, color='blue', label="Learned (A)")
ax1.plot(slip_angle_list, Fy_B_list, color='darkred', label="Learned (B)")

ax1.grid()
ax1.set(title='Lateral Tire Model (A, B)', xlabel='Side Slip Angle [rad]', ylabel='Lateral Tire Force [kN]')
ax1.set(ylim=[-5,5])
ax1.set(xlim=[-0.35,0.35])
ax1.legend()
plt.show()