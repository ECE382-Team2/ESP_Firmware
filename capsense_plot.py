import time  
import numpy as np 
from collections import deque  # for fixed-length queues
# import serial  # uncomment for real board  

NUM_SENSORS = 4      # number of sensors in a message
MAX_ENTRIES = 5    # maximum number of rows to store in each matrix

# ================================================================ 
# Assuming the following data format from serial:  
# Each line: mode,sensor1,sensor2,...,sensorN 
# Expected example lines: 0,12,34,56,78  (normal) 
#                         1,23,45,67,89  (shear)  
# ================================================================ 
# fake serial input for testing  
class FakeSerial:  
    def __init__(self, num_sensors = NUM_SENSORS): 
        self.counter = 0    # to alternate between shear and normal
        self.num_sensors = num_sensors  # store number of sensors for random generation

    def readline(self):  
        # generate fake data; random integers between 1 and 100  # create sample values
        shear_values = np.random.randint(1, 100, self.num_sensors) 
        normal_values = np.random.randint(1, 100, self.num_sensors) 

        # mode: 1 for shear, 0 for normal 
        if self.counter % 2 == 0:  
            line = "1," + ",".join(map(str, shear_values)) + "\n" # counter is even (shear)
        else:  # odd counter -> normal mode
            line = "0," + ",".join(map(str, normal_values)) + "\n"  # coutnr is odd (normal) 

        self.counter += 1  
        time.sleep(0.25)                # delay 
        return line.encode('utf-8')     # return as bytes 

# ================================================================ 
# setup serial connection  

# ser = serial.Serial('COM4', 115200, timeout=1)  #  com port, baud rate, timeout 
ser = FakeSerial(NUM_SENSORS)  

# data matrices with max length  
normal_matrix = deque(maxlen=MAX_ENTRIES)  
shear_matrix = deque(maxlen=MAX_ENTRIES) 

print("Waiting for data...")  

while True: 
    line = ser.readline().decode('utf-8').strip()  # strip whitespace and newline
    if not line:  # skip if empty
        continue 

    try: 
        parts = line.split(',')  # split the into parts by comma
        mode = int(parts[0].strip())  # 0 (shear) or 1 (normal)
        values = np.array(list(map(float, parts[1:])))  # capacitance values to array

        # skip if not the expected number of sensors
        if len(values) != NUM_SENSORS:  
            continue 

        if mode == 0:  # normal mode
            normal_matrix.append(values) 
        elif mode == 1:  # shear mode
            shear_matrix.append(values)  

        # convert queues to arrays
        # arrays might be needed to use visualization libraries 
        normal_array = np.array(normal_matrix)  
        shear_array = np.array(shear_matrix)  

        print("Normal matrix:\n", normal_array)  
        print("Shear matrix:\n", shear_array, "\n")  

    except Exception as e:  
        print(f"Error parsing line: {line} -> {e}") 
