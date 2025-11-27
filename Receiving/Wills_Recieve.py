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
        # Head 1: processes first 8 elements + previous outputs, outputs fx, tx, ty
        self.head1_fc1 = nn.Linear(8 + 3, 32)  # 8 inputs + 3 previous outputs (fx, tx, ty)
        self.head1_fc2 = nn.Linear(32, 32)
        self.head1_fc3 = nn.Linear(32, 32)
        self.head1_fc4 = nn.Linear(32, 3)  # outputs [fx, tx, ty]
        
        # Head 2: processes second 8 elements + previous outputs, outputs fz, fy, tz
        self.head2_fc1 = nn.Linear(8 + 3, 32)  # 8 inputs + 3 previous outputs (fz, fy, tz)
        self.head2_fc2 = nn.Linear(32, 32)
        self.head2_fc3 = nn.Linear(32, 32)
        self.head2_fc4 = nn.Linear(32, 3)  # outputs [fz, fy, tz]
        
        self.relu = nn.ReLU()

        # Previous outputs - will be expanded to match batch size
        self.prev_head1_out = None  # Previous [fx, tx, ty]
        self.prev_head2_out = None  # Previous [fz, fy, tz]

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Initialize previous outputs if needed (first forward pass)
        if self.prev_head1_out is None or self.prev_head1_out.shape[0] != batch_size:
            self.prev_head1_out = torch.zeros(batch_size, 3, device=x.device)
            self.prev_head2_out = torch.zeros(batch_size, 3, device=x.device)
        
        # Split input: first 8 elements and second 8 elements
        x1 = x[:, :8]   # First 8 elements
        x2 = x[:, 8:]   # Second 8 elements

        # Head 1 forward pass (with previous fx, tx, ty)
        h1_input = torch.cat([x1, self.prev_head1_out], dim=1)
        h1 = self.relu(self.head1_fc1(h1_input))
        h1 = self.relu(self.head1_fc2(h1))
        h1 = self.relu(self.head1_fc3(h1))
        h1_out = self.head1_fc4(h1)  # [fx, tx, ty]
        
        # Head 2 forward pass (with previous fz, fy, tz)
        h2_input = torch.cat([x2, self.prev_head2_out], dim=1)
        h2 = self.relu(self.head2_fc1(h2_input))
        h2 = self.relu(self.head2_fc2(h2))
        h2 = self.relu(self.head2_fc3(h2))
        h2_out = self.head2_fc4(h2)  # [fz, fy, tz]
        
        # Combine outputs to form [fx, fy, fz, tx, ty, tz]
        fx = h1_out[:, 0:1]
        tx = h1_out[:, 1:2]
        ty = h1_out[:, 2:3]
        fz = h2_out[:, 0:1]
        fy = h2_out[:, 1:2]
        tz = h2_out[:, 2:3]
        
        output = torch.cat([fx, fy, fz, tx, ty, tz], dim=1)

        # Store current outputs for next forward pass (detach to avoid gradients)
        self.prev_head1_out = h1_out.detach()
        self.prev_head2_out = h2_out.detach()
        
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
poly_model = pickle.load(open('ESP_Firmware/calibration/models.pkl', 'rb'))
poly = PolynomialFeatures(degree=2, include_bias=True)

def polynomial_predict(X):
    # Load the polynomial models
    poly_features = poly.fit_transform(X.reshape(1, -1))

    # Predict forces and torques
    output = np.zeros(6)
    force_torque_names = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']

    for i, name in enumerate(force_torque_names):
        if name in poly_model:
            output[i] = poly_model[name].predict(poly_features)[0]

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
        # return polynomial_predict(biased_data)
        return nn_predict(biased_data)
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
    ax.set_ylim(-1000, 1000)       # calibrated FT values
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
