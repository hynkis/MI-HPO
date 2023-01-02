import math
import numpy as np

def lateral_tire_model(slip_angle, params):
    """
    Lateral Tire Model Identification (Pacejka tire model)
    (numpy vectorization for fast computation)
        F_y = Sy + D * np.sin(C * np.arctan2(B * np.deg2rad(slip_angle + Sx)))
        @ Input
            - slip_angle (rad)

        @ Parameters [params]
            - D : pacejka tire model parameter
            - C : pacejka tire model parameter
            - B : pacejka tire model parameter
            - Sx : offset parameter (x-axis)
            - Sy : offset parameter (y-axis)
    """
    # Parameters
    B, C, D, Sx, Sy = params[0], params[1], params[2], params[3], params[4]

    # Lateral Tire Force (slip_angle of data is flipped)
    F_y = Sy + D * np.sin(C * np.arctan2(B * -(slip_angle + Sx), 1))

    return F_y
