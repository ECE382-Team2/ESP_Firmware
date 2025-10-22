import time
import numpy as np
import matplotlib.pyplot as plt
import serial
import joblib


# ==============================
# Derivative shit that hopes the sensativity of the sensor is consistent
# ==============================
# Previous capacitance values for derivative calculation
C1_prev = 0
C2_prev = 0
C3_prev = 0
C4_prev = 0

# Accumulated values from derivatives
C1_accum = 0
C2_accum = 0
C3_accum = 0
C4_accum = 0

first_run = True    

def update_derivatives(C_current):
    """
    Compute derivatives and update accumulated values.
    
    Args:
        C_current: numpy array of current capacitance values [C1, C2, C3, C4]
    
    Returns:
        numpy array of accumulated values
    """
    global C1_prev, C2_prev, C3_prev, C4_prev
    global C1_accum, C2_accum, C3_accum, C4_accum
    global first_run

    if first_run:
        C1_prev, C2_prev, C3_prev, C4_prev = C_current
        first_run = False

    # Calculate derivatives (change from previous reading)
    dC1 = C_current[0] - C1_prev
    dC2 = C_current[1] - C2_prev
    dC3 = C_current[2] - C3_prev
    dC4 = C_current[3] - C4_prev
    
    # Accumulate the changes
    C1_accum += dC1
    C2_accum += dC2
    C3_accum += dC3
    C4_accum += dC4
    
    # Update previous values for next iteration
    C1_prev = C_current[0]
    C2_prev = C_current[1]
    C3_prev = C_current[2]
    C4_prev = C_current[3]
    
    return np.array([C1_accum, C2_accum, C3_accum, C4_accum])

NUM_SENSORS = 4
max_points = 100  # how many points to show on plot

# ==============================
# Ask user whether to use regression
# ==============================
use_model_input = input("Use calibration model? (y/n): ").strip().lower()
use_model = use_model_input == 'y'

# ==============================
# Load models if needed
# ==============================
if use_model:
    normal_model = joblib.load("normal_data_linear_model.pkl")
    shear_model = joblib.load("shear_data_linear_model.pkl")
    print("Loaded Normal and Shear models.")

# ==============================
# Setup serial connection
# ==============================
class FakeSerial:
    def __init__(self):
        self.counter = 0
    def readline(self):
        mode = self.counter % 2
        values = np.random.randint(1, 100, NUM_SENSORS)
        self.counter += 1
        time.sleep(0.1)
        return (f"{mode}," + ",".join(map(str, values)) + "\n").encode("utf-8")

try:
    ser = serial.Serial("COM5", 9600, timeout=1)
except:
    print("Using FakeSerial() for testing")
    ser = FakeSerial()

# ==============================
# Setup plot
# ==============================
plt.ion()
fig, ax = plt.subplots()

# Set y-limits based on mode
if use_model:
    ax.set_ylim(-10, 10)       # calibrated FT values
else:
    ax.set_ylim(-20000, 10000)        # raw capacitance values

ax.set_xlim(0, max_points)
ax.set_xlabel("Sample")
ax.set_ylabel("Force / Torque / Capacitance")

# Choose plot keys
if use_model:
    plot_keys = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
else:
    plot_keys = ["C1", "C2", "C3", "C4"]

lines = {}
for key in plot_keys:
    lines[key] = ax.plot([], [], label=key)[0]
ax.legend()
history = {key: [] for key in plot_keys}

# ==============================
# Main loop
# ==============================
try:
    while True:
        line = ser.readline().decode("utf-8").strip()
        if not line:
            continue

        try:
            parts = line.split(",")
            mode_flag = int(parts[0])
            C = np.array(list(map(float, parts[1:]))).reshape(1, -1)
            if C.shape[1] != NUM_SENSORS:
                continue
        except:
            continue

        # Determine data to plot
        if use_model:
            if mode_flag == 0:
                data = normal_model.predict(C)[0]
            else:
                data = shear_model.predict(C)[0]
        else:
            if mode_flag == 0:
                data = C[0]  # raw capacitance values
                # Will's Derivative stuff
                data = update_derivatives(data)

        # Update history
        for i, key in enumerate(plot_keys):
            history[key].append(data[i])
            if len(history[key]) > max_points:
                history[key].pop(0)

        # Update plot
        for key in plot_keys:
            lines[key].set_data(range(len(history[key])), history[key])
        ax.set_xlim(0, max(len(history[plot_keys[0]]), max_points))
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\nStopped real-time visualization")
