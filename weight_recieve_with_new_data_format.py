import time
import numpy as np
import matplotlib.pyplot as plt
import serial
import joblib

# ==============================
# Derivative shit that hopes the sensativity of the sensor is consistent
# ==============================
# Previous capacitance values for derivative calculation
global C_init
global first_run

NUM_SENSORS = 8
C_init = np.zeros(NUM_SENSORS)   # ensure length matches NUM_SENSORS
first_run = True

max_points = 100  # how many points to show on plot

# ==============================
# Sensor Weights
# ==============================
# Adjust these numbers to scale each sensor’s contribution
sensor_weights = np.array([
    [3.046183461811196e-07, 3.915173383998768e-01, 4.919275893827008e-01, 1.814119519290021e-01, -7.229482869826916e-03, -1.815187667211892e+00, 2.737244577593229e-01, -5.299132964183279e-01],
    [2.305747385103482e-06, 1.953131646367295e-02, 2.351761635323825e-01, 7.055264424244227e-02, -2.410359779319578e-02, -1.290470723448289e+01, 1.831255948179031e+00, 3.597199040086411e-01],
    [1.891128009663307e-05, -7.398897508170579e-01, -8.370138753089616e+00, 7.612665051483160e-01, 3.384519456483315e-01, -1.158865106260597e+02, -2.076698839480769e+01, 4.635287877579364e+00],
    [1.498639606169316e-07, -4.710875444588795e-02, 4.960105440245996e-02, 3.114757774792658e-03, 4.906874153724602e-03, -9.123405020614327e-01, -1.046435418353015e-01, 7.507823714662149e-03],
    [-3.054778366279614e-08, 4.080036379369498e-03, 6.455896773566308e-02, -2.519378718476815e-03, 5.213285370174100e-04, 2.999187441123031e-01, 4.016832552392534e-01, -2.914926456904444e-02],
    [3.782240326022936e-09, 1.023246771070913e-02, 1.174522024606123e-02, 2.776361865601717e-03, -2.239223780277097e-03, -3.233282603388142e-02, -2.151158198875897e-02, 3.964280497141005e-03],
])

sensor_biases = np.array([
    -6.018745857180133e+04,  # fx_intercept
    -2.016358749653060e+04,  # fy_intercept
    4.119084927626266e+05,   # fz_intercept
    -8.804094328781720e+02,  # tx_intercept
    -3.610360020036960e+03,  # ty_intercept
    -1.284116167795461e+03   # tz_intercept
])
print(f"Using fixed sensor weights: {sensor_weights}")

# ==============================
# Ask user whether to use regression
# ==============================
use_model_input = input("Use calibration model? (y/n): ").strip().lower()
use_model = use_model_input == 'y'

# ==============================
# Load models if needed
# ==============================
if use_model:
    # normal_model = joblib.load("normal_data_linear_model.pkl")
    # shear_model = joblib.load("shear_data_linear_model.pkl")
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
    ser = serial.Serial("COM5", 115200, timeout=1)
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
    ax.set_ylim(-10000, 10000)       # calibrated FT values
else:
    ax.set_ylim(-20000, 10000) # raw capacitance values

ax.set_xlim(0, max_points)
ax.set_xlabel("Sample")
ax.set_ylabel("Force / Torque / Capacitance")

# Choose plot keys (make these depend on NUM_SENSORS)
if use_model:
    plot_keys = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
else:
    # For 8 sensors keep previous naming, otherwise generate generic names
    if NUM_SENSORS == 8:
        plot_keys = ["C1", "C2", "C3", "C4", "S1", "S2", "S3", "S4"]
    else:
        plot_keys = [f"C{i+1}" for i in range(NUM_SENSORS)]

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
            vals = list(map(float, parts[1:]))
            if len(vals) != NUM_SENSORS:
                # skip malformed lines
                continue
            C = np.array(vals).reshape(1, -1)
            if first_run:
                C_init = C[0].copy()
                first_run = False
        except:
            continue

        # Determine data to plot
        if use_model:
            # pred = normal_model.predict(C)
            # data = np.array(pred[0])  # expect length 6
            data = sensor_weights @ (C[0] - C_init) + sensor_biases
        else:
            # raw capacitance delta -> length NUM_SENSORS
            data = (C[0] - C_init)

        # Update history safely (handle mismatched lengths)
        n_plot = len(plot_keys)
        n_data = len(data)
        for i, key in enumerate(plot_keys):
            if i < n_data:
                history[key].append(data[i])
            else:
                # no corresponding data channel; append 0 so plot lines stay aligned
                history[key].append(0.0)
            if len(history[key]) > max_points:
                history[key].pop(0)

        # Update plot
        for key in plot_keys:
            lines[key].set_data(range(len(history[key])), history[key])
        ax.set_xlim(0, max(len(history[plot_keys[0]]), max_points))
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\nStopped real-time visualization")
