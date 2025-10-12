import time
import os
import numpy as np
import serial

NUM_SENSORS = 4      # number of sensors in a message
CSV_HEADERS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "C1", "C2", "C3", "C4"]

# ================================================================
# Fake serial for testing
# ================================================================
class FakeSerial:
    def __init__(self, num_sensors=NUM_SENSORS):
        self.num_sensors = num_sensors

    def readline(self):
        # Random capacitance data
        values = np.random.randint(1, 100, self.num_sensors)
        time.sleep(0.25)
        return ",".join(map(str, values)).encode("utf-8")

# ================================================================
# Setup serial connection
# ================================================================
try:
    ser = serial.Serial('COM6', 9600, timeout=1)
    print("Connected to COM6")
except Exception as e:
    print(f"Could not open COM6: {e}\nUsing FakeSerial() instead.")
    ser = FakeSerial(NUM_SENSORS)

# ================================================================
# CSV setup — write header only if file does not exist
# ================================================================
def ensure_csv_exists(filename, mode_label):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(f"{mode_label}\n")
            f.write("FT Values,,,,,,Electrode Values\n")
            f.write(",".join(CSV_HEADERS) + "\n")

# ================================================================
# User inputs
# ================================================================
mode_input = input("\nEnter mode ('normal' or 'shear'): ").strip().lower()
if mode_input not in ["normal", "shear"]:
    raise ValueError("Mode must be 'normal' or 'shear'")

N = int(input("\nEnter number of samples to collect: "))

print("\nEnter measured FT values (these will apply to all samples):")
Fx = float(input("Fx: "))
Fy = float(input("Fy: "))
Fz = float(input("Fz: "))
Tx = float(input("Tx: "))
Ty = float(input("Ty: "))
Tz = float(input("Tz: "))

filename = f"{mode_input}_data.csv"
ensure_csv_exists(filename, f"{mode_input.upper()} FT DATA")

print(f"\nCollecting {N} samples for {mode_input.upper()} mode with same FT values...")

# ================================================================
# Data collection loop
# ================================================================
for i in range(N):
    # Read capacitance values from serial
    line = ser.readline().decode("utf-8").strip()
    capacitance_values = np.array(list(map(float, line.split(','))))

    if len(capacitance_values) != NUM_SENSORS:
        print(f"Warning: received wrong number of capacitance values for sample {i+1}. Skipping.")
        continue

    # Combine FT + capacitance into one row
    full_row = np.concatenate(([Fx, Fy, Fz, Tx, Ty, Tz], capacitance_values))

    # Append to CSV
    with open(filename, "a") as f:
        np.savetxt(f, full_row.reshape(1, -1), delimiter=",", fmt="%.3f")

print(f"\nData collection complete. Appended {N} samples to {filename}")
print("===============================================================")
