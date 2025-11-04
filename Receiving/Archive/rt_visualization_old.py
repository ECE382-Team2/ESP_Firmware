import time
import numpy as np
import matplotlib.pyplot as plt
import serial
import joblib

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
    ser = serial.Serial("COM14", 9600, timeout=1)
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
    ax.set_ylim(0, 110)        # raw capacitance values

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
            data = C[0]  # raw capacitance values

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
