import numpy as np

# Max solenoid current in Milliamps
max_ma = 1700.0

# State Keys
state_keys = [
    "Rod_Pressure",
    "Base_Pressure",
    "System_Pressure",
    "Load_Sense_Pressure",
    "Reservoir_Temperature",
    "Height",
    "Height_Rate",
]
# Control Keys : this is temporary till new dynamics are found in same order
control_keys = [
    "Height",
    "Rod_Pressure",
    "Base_Pressure",
    "System_Pressure",
    "Load_Sense_Pressure",
    "Reservoir_Temperature",
    "Height_Rate",
]

# Set operating height limits
height_max = 0.8
height_min = 0.3
goal_max = 0.75
goal_min = 0.35

# LQR control matrix K
K = np.array(
    [
        [
            -1615.04751531,
            341.3410131,
            345.0694559,
            -552.2149965,
            158.71363505,
            -28935.39623854,
            -1835.66716584,
        ]
    ]
)


# action limits
action_high = np.array([max_ma], dtype=np.float32)
action_low = -action_high
