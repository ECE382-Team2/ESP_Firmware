import time
import os
import serial
import re

# SERIAL_PORT = '/dev/ttyACM0'  # serial port to connect to
SERIAL_PORT = 'COM7'  # serial port to connect to
CSV_HEADERS = ["Value", "Mode", "Port", "Timestamp Sent", "Timestamp Received"]


# ================================================================
# Setup serial connection
# ================================================================
try:
    ser = serial.Serial(SERIAL_PORT, baudrate=921600,timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"{e}\nCannot find serial port {SERIAL_PORT}")
    exit()


# ================================================================
# CSV setup — write header only if file doesn't exist or is empty
# ================================================================
def ensure_csv_exists(filename):
    # Check if file exists and is not empty
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, "w") as f:
            f.write(",".join(CSV_HEADERS) + "\n")

timestamp_str = time.strftime("%Y%m%d_%H%M%S")
scenario = input("Enter test scenario: ").strip()
scenario = re.sub(r'[^\w\-_\.]', '_', scenario)  # Replace special chars with _
filename = f"{timestamp_str}_{scenario}_data.csv"
ensure_csv_exists(filename)

N = int(input("\nEnter duration in seconds to collect data: "))

print(f"\nCollecting data for {N} seconds...")

# ================================================================
# Data collection loop
# ================================================================
start_time = time.time()
samples_collected = 0
ser.readline()

while time.time() - start_time < N:
    try:
        elapsed = time.time() - start_time
        time_left = N - elapsed
        print(f"\rElapsed: {elapsed:.1f}s | Time left: {time_left:.1f}s | Samples: {samples_collected}", end='', flush=True)
        # Read capacitance values from serial
        line = ser.readline().decode("utf-8").strip()
        if (len(line) < 4): continue
        timestamp = time.time() * 1000
        try:
            capacitance_values = [float(x) for x in line.split(',')]
        except Exception as e:
            print(line, e)

        if len(capacitance_values) != 4:
            print(f"Warning: received wrong number of capacitance values for sample {samples_collected + 1}. Skipping.")
            continue

        samples_collected += 1

        # Create row with 4 values + timestamp
        full_row = capacitance_values + [timestamp]

        # Append to CSV
        with open(filename, "a") as f:
            f.write(','.join(str(x) for x in full_row) + '\n')
    except Exception as e:
        print('uhhh idk what hapen', e)

print(f"\nData collection complete. Collected {samples_collected} samples in {N} seconds to {filename}")
print("===============================================================")
