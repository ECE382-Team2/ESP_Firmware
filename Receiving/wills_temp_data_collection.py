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
    ser = serial.Serial('COM7', 115200, timeout=1)
    print("Connected to COM7")
except Exception as e:
    print(f"{e}\nUsing FakeSerial() instead")
    ser = FakeSerial(NUM_SENSORS)

# ================================================================
# CSV setup — write header only if file doesn't exist or is empty
# ================================================================
def ensure_csv_exists(filename, mode_label):
    # Check if file exists and is not empty
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, "w") as f:
            f.write(f"{mode_label}\n")
            f.write("FT Values,,,,,,Electrode Values\n")
            f.write(",".join(CSV_HEADERS) + "\n")

# ================================================================
# User inputs
# ================================================================
# mode_input = input("\nEnter mode ('normal' or 'shear'): ").strip().lower()
# if mode_input not in ["normal", "shear"]:
#     raise ValueError("Mode must be 'normal' or 'shear'")

N = int(input("\nEnter number of samples to collect: "))

######## uncomment if you want to manually input FT values ########

# print("\nEnter measured FT values (these will apply to all samples):")
# Fx = float(input("Fx: "))
# Fy = float(input("Fy: "))
# Fz = float(input("Fz: "))
# Tx = float(input("Tx: "))
# Ty = float(input("Ty: "))
# Tz = float(input("Tz: "))

######## uncomment if you want to manually input FT values ########

# uncomment if you collect FT data outside this script
if(N>-1):
    A = input("Hit a button when you are ready to go ")

filename = f"data.csv"
ensure_csv_exists(filename, "FT DATA")

print(f"\nCollecting {N} samples")

# ================================================================
# Data collection loop
# ================================================================
for i in range(N):
    # Read capacitance values from serial
    mode_bit = 1

    # waits for normal mode
    while mode_bit != 0:
        line = ser.readline().decode("utf-8").strip()
        try:
            capacitance_values = np.array(list(map(float, line.split(','))))
            mode_bit = int(capacitance_values[0])
        except:
            continue
    
    # # # checks for correct number of sensors
    # if len(capacitance_values) != NUM_SENSORS + 1:
    #     print(f"Warning: received wrong number of capacitance values for sample {i+1}. Skipping.")
    #     continue

    # Combine FT with capacitance into one row
    # full_row = np.concatenate(([Fx, Fy, Fz, Tx, Ty, Tz], capacitance_values[1:])) # when you want to manually input FT values
    full_row = capacitance_values[1:]


    # waits for shear mode
    while mode_bit != 1:
        line = ser.readline().decode("utf-8").strip()
        try:
            capacitance_values = np.array(list(map(float, line.split(','))))
            mode_bit = int(capacitance_values[0])
        except:
            continue

    # final concatination and write
    # Append timestamp (seconds since epoch) to the row
    timestamp = time.time()
    full_row = np.concatenate((full_row, capacitance_values[1:], [timestamp]))

    # Append to CSV
    with open(filename, "a") as f:
        np.savetxt(f, full_row.reshape(1, -1), delimiter=",", fmt="%.3f")

print(f"\nData collection complete. Appended {N} samples to {filename}")
print("===============================================================")
