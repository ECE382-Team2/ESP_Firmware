import pickle
import time
from xml.parsers.expat import model
import numpy as np
import matplotlib.pyplot as plt
import serial
import joblib
from collections import deque

from sklearn.preprocessing import PolynomialFeatures
import torch
import torch.nn as nn

# Define the neural network with two heads
class ForceNet(nn.Module):
    def __init__(self):
        super(ForceNet, self).__init__()
        # Single network: processes all 16 inputs + previous 6 outputs + previous 16 capacitance
        self.fc1 = nn.Linear(16 + 6 + 16, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 6)  # outputs [fx, fy, fz, tx, ty, tz]

        self.relu = nn.ReLU()

        # Previous output and capacitance values
        self.prev_output = None
        self.prev_cap = None

    def forward(self, x):
        batch_size = x.shape[0]

        # Initialize previous output if needed (first forward pass)
        if self.prev_output is None or self.prev_output.shape[0] != batch_size:
            self.prev_output = torch.zeros(batch_size, 6, device=x.device)

        # Initialize previous capacitance if needed (first forward pass)
        if self.prev_cap is None or self.prev_cap.shape[0] != batch_size:
            self.prev_cap = torch.zeros(batch_size, 16, device=x.device)

        # Concatenate current input with previous output and previous capacitance
        combined_input = torch.cat([x, self.prev_output, self.prev_cap], dim=1)

        # Forward pass through the network
        h = self.relu(self.fc1(combined_input))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        output = self.fc4(h)

        # Store current output for next forward pass (detach to avoid gradients)
        self.prev_output = output.detach()

        # Store current capacitance for next forward pass (detach to avoid gradients)
        self.prev_cap = x.detach()

        return output
    
nn_model = ForceNet()
# Note: this path only works if your working directory is Tactile. Otherwise, you might need to delete the ESP_Firmware part
nn_model.load_state_dict(torch.load('ESP_Firmware/calibration/nn_model', weights_only=True))
nn_model.eval()

def nn_predict(X):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
        output_tensor = nn_model(input_tensor)
        output = output_tensor.squeeze(0).numpy()  # Remove batch dimension
    return output  # Fx, Fy, Fz, Tx, Ty, Tz


######### polynomial regression approach ##########
poly_model = pickle.load(open('ESP_Firmware/calibration/poly_model.pkl', 'rb'))

def polynomial_predict(X):
    # Predict forces and torques
    # Note: poly_model[name] is a Pipeline that includes PolynomialFeatures + LinearRegression
    # So we pass the raw input directly without manual transformation
    output = np.zeros(6)
    force_torque_names = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']

    X_reshaped = X.reshape(1, -1)  # Reshape to (1, 16) for single prediction
    
    for i, name in enumerate(force_torque_names):
        if name in poly_model:
            output[i] = poly_model[name].predict(X_reshaped)[0]

    return output


# Here is where all of the processing happens before you send data out
first_run = True
def process(data, use_model=False):

    global first_run 
    global biases

    if (first_run):
        first_run = False
        biases = data.copy()

    biases = 0.999 * biases + 0.001 * data  # Update biases with moving average

    biased_data = (data - biases)

    if use_model:
        # result = polynomial_predict(biased_data)
        result = nn_predict(biased_data)
        # result[2] = 0  # Zero out Fz (index 2: Fx=0, Fy=1, Fz=2, Tx=3, Ty=4, Tz=5)
        return result
    else:
        return biased_data

NUM_SENSORS = 16
max_points = 100  # how many points to show on plot

# ==============================
# Ask user whether to use regression
# ==============================
use_model_input = input("Use calibration model? (y/n): ").strip().lower()
use_model = use_model_input == 'y'

# ==============================
# Load models if needed
# ==============================
# if use_model:
#     normal_model = joblib.load("normal_data_linear_model.pkl")
#     shear_model = joblib.load("shear_data_linear_model.pkl")
#     print("Loaded Normal and Shear models.")

# ==============================
# Setup plot
# ==============================
plt.ion()
fig, ax = plt.subplots()

# Set y-limits based on mode
if use_model:
    ax.set_ylim(-100, 100)       # calibrated FT values
else:
    ax.set_ylim(-20000, 10000)        # raw capacitance values

ax.set_xlim(0, max_points)
ax.set_xlabel("Sample")
ax.set_ylabel("Force / Torque / Capacitance")

# Choose plot keys
if use_model:
    plot_keys = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
else:
    plot_keys = ["C1", "C2", "C3", "C4","C5", "C6", "C7", "C8","C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16"]

lines = {}
for key in plot_keys:
    lines[key] = ax.plot([], [], label=key)[0]
ax.legend()
history = {key: deque(maxlen=max_points) for key in plot_keys}

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
    ser = serial.Serial("COM5", 115200, timeout=1)
    # ser.reset_input_buffer()
except:
    print("Using FakeSerial() for testing")
    ser = FakeSerial()

# ==============================
# Main loop
# ==============================
try:

    # # Warm start for reading data
    # for _ in range(10):
    #     ser.reset_input_buffer()
    #     line = ser.readline().decode("utf-8").strip()

    while True:
        ser.reset_input_buffer()
        ser.readline() # Discard one line to avoid partial reads
        line = ser.readline().decode("utf-8").strip()
        if not line:
            continue

        try:
            parts = line.split(",")
            # mode_flag = int(parts[0])
            C = np.array(list(map(int, parts))).reshape(1, -1)
            
            if C.shape[1] != NUM_SENSORS:
                continue
        except:
            continue

        # Determine data to plot
        # if use_model:
        #     data = normal_model.predict(C)[0]
        # else:
        data = process(C[0], use_model=use_model)  # raw capacitance values

        # Update history
        for i, key in enumerate(plot_keys):
            history[key].append(data[i])

        # Update plot
        for key in plot_keys:
            lines[key].set_data(range(len(history[key])), history[key])
        ax.set_xlim(0, max(len(history[plot_keys[0]]), max_points))
        plt.pause(0.000001)

except KeyboardInterrupt:
    print("\nStopped real-time visualization")
