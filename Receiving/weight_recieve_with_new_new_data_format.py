import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import serial
import serial.tools.list_ports
# import joblib

def detect_serial_port():
    """
    Auto-detect available serial ports and return the most likely candidate.
    Returns the port name or prompts user to select if multiple ports found.
    """
    ports = list(serial.tools.list_ports.comports())
    
    # Filter out ttyS ports (not typically used for ESP devices)
    ports = [port for port in ports if not port.device.startswith('/dev/ttyS')]
    
    if not ports:
        print("No serial ports found!")
        return None
    
    print(f"Found {len(ports)} serial port(s):")
    for i, port in enumerate(ports):
        print(f"  {i+1}: {port.device} - {port.description}")
    
    if len(ports) == 1:
        # Only one port, use it automatically
        selected_port = ports[0].device
        print(f"Auto-selected: {selected_port}")
        return selected_port
    else:
        # Multiple ports, let user choose
        while True:
            try:
                choice = input(f"Select port (1-{len(ports)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(ports):
                    selected_port = ports[idx].device
                    print(f"Selected: {selected_port}")
                    return selected_port
                else:
                    print(f"Please enter a number between 1 and {len(ports)}")
            except (ValueError, EOFError):
                print("Invalid input. Please enter a number.")

# ==============================
# Auto-detect serial port
# ==============================
SERIAL_PORT = detect_serial_port()
if SERIAL_PORT is None:
    print("No serial port available. Exiting.")
    exit()

print(f"Using serial port: {SERIAL_PORT}")

# ==============================
# predicition fucntion
# ==============================

# ==============================
# predicition fucntion
# ==============================

def subtraction_shear(data_map):
    """
    Convert shear data from individual ports (1a,1b,2a,2b,etc.) to combined ports (1,2,3,4)
    by subtracting pairs (1a-1b, 2a-2b, etc.) and averaging timestamps.
    Normal mode data passes through unchanged.
    """
    processed_map = {}
    
    for mode, ports in data_map.items():
        if mode == 'shear':
            # Process shear mode: combine pairs
            processed_map[mode] = {}
            
            # Define pairs: 1a-1b=1, 2a-2b=2, 3a-3b=3, 4a-4b=4
            pairs = [('1a', '1b', '1'), ('2a', '2b', '2'), ('3a', '3b', '3'), ('4a', '4b', '4')]
            
            for port_a, port_b, result_port in pairs:
                if port_a in ports and port_b in ports:
                    data_a = ports[port_a]
                    data_b = ports[port_b]
                    
                    # Find the minimum length to match by index
                    min_len = min(len(data_a['timestamps']), len(data_b['timestamps']))
                    
                    if min_len > 0:
                        # Calculate subtracted values and averaged timestamps
                        subtracted_values = []
                        averaged_timestamps = []
                        
                        for i in range(min_len):
                            # Subtract values: a - b
                            subtracted_values.append(data_a['values'][i] - data_b['values'][i])
                            # Average timestamps
                            averaged_timestamps.append((data_a['timestamps'][i] + data_b['timestamps'][i]) / 2.0)
                        
                        # Store the processed data
                        processed_map[mode][result_port] = {
                            'timestamps': averaged_timestamps,
                            'values': subtracted_values
                        }
        else:
            # Pass through other modes (like 'normal') unchanged
            processed_map[mode] = ports.copy()
    
    return processed_map
    

post_process = subtraction_shear


# ==============================
# Derivative shit that hopes the sensativity of the sensor is consistent
# ==============================
# Previous capacitance values for derivative calculation
global sensor_data_map

# Track when to update plot structure (after 2s, then every .05s)
last_plot_update = time.time() + 2.5
plot_update_interval = 0.09 + 0.01 # seconds - reduced from 1.05 for faster updates
max_time = 10  # how many seconds to show on plot



# Global map: mode -> port -> {'timestamps': [], 'values': []}
sensor_data_map = {}

# Track last absolute timestamp to convert deltas to absolute
last_absolute_timestamp = 0  # Initialize in seconds

# Mapping functions
def map_mode(mode_num):
    """Map mode number to name: 0 -> 'normal', 1 -> 'shear'"""
    mode_map = {0: 'normal', 1: 'shear'}
    return mode_map.get(int(mode_num), f'unknown_{mode_num}')

def map_port(port_num):
    """Map port number to name: 0-7 -> '1a','1b','2a','2b','3a','3b','4a','4b'"""
    port_map = {
        0: '1a', 1: '1b', 
        2: '2a', 3: '2b',
        4: '3a', 5: '3b',
        6: '4a', 7: '4b'
    }
    return port_map.get(int(port_num), f'unknown_{port_num}')

def parse_line(line):
    """
    Parse a line of serial data into value, mode, port, timestamp.
    Returns tuple of (value, mode_str, port_str, timestamp) or None if parsing fails.
    """
    try:
        parts = line.split(",")
        if len(parts) != 4:
            return None
        
        timestamp = float(parts[0]) / 1e4 # sensor provides deci milli second
        if timestamp < 0:
            timestamp += 1
        mode_str = map_mode(parts[1])
        port_str = map_port(parts[2])
        value = float(parts[3])
        
        return (timestamp, mode_str, port_str, value)
    except (ValueError, IndexError):
        return None
# # ==============================
# # Ask user whether to use regression
# # ==============================
# use_model_input = input("Use calibration model? (y/n): ").strip().lower()
# use_model = use_model_input == 'y'

try:
    ser = serial.Serial(SERIAL_PORT, baudrate=921600, timeout=1)
except:
    print("No Serial")
    exit()

# ==============================
# Setup plot
# ==============================
plt.ion()
# Force the plot window to show
plt.show(block=False)
fig = None  # Initialize as None instead of dict
axes_dict = {}
lines_dict = {}  # mode -> port -> line object
text_dict = {}  # mode -> text object
checkbox_dict = {}  # mode -> CheckButtons widget
visibility_dict = {}  # mode -> port -> boolean (True = visible)
checkbox_axes_dict = {}  # mode -> checkbox axes

def on_checkbox_clicked(label, mode):
    """Callback function for checkbox clicks"""
    port = label  # The label is the port name
    visibility_dict[mode][port] = not visibility_dict[mode][port]
    
    # Update line visibility
    if port in lines_dict.get(mode, {}):
        lines_dict[mode][port].set_visible(visibility_dict[mode][port])
    
    # Force redraw
    if fig:
        fig.canvas.draw()

def update_plot_structure(data_map):
    """Dynamically create/update plot structure based on current modes in sensor_data_map"""
    global fig, axes_dict, lines_dict, text_dict, checkbox_dict, visibility_dict, checkbox_axes_dict
    
    modes_in_data = list(data_map.keys())
    
    # If no modes yet, skip
    if len(modes_in_data) == 0:
        return
    
    # Check if we need to recreate the figure
    modes_changed = set(modes_in_data) != set(axes_dict.keys())
    
    # Check if ports have changed for existing modes
    ports_changed = False
    for mode in modes_in_data:
        if mode in axes_dict:
            current_ports = set(data_map[mode].keys())
            plotted_ports = set(lines_dict.get(mode, {}).keys())
            if current_ports != plotted_ports:
                ports_changed = True
                break
    
    if modes_changed or ports_changed or fig is None:
        if fig is not None:
            fig.clf()  # Clear current figure object instead of closing window
            fig.set_size_inches(12, 4 * len(modes_in_data))  # Increased width for checkboxes
        else:
            # First time creating figure
            fig = plt.figure(figsize=(12, 4 * len(modes_in_data)))  # Increased width for checkboxes
        
        axes_dict = {}
        lines_dict = {}
        text_dict = {}
        checkbox_dict = {}
        visibility_dict = {}
        checkbox_axes_dict = {}
        
        for idx, mode in enumerate(sorted(modes_in_data)):
            # Create main plot axes (takes up most of the width)
            ax = fig.add_subplot(len(modes_in_data), 10, idx * 10 + 1)
            ax.set_position([0.1, 0.1 + idx * (0.8 / len(modes_in_data)), 0.65, 0.7 / len(modes_in_data)])
            axes_dict[mode] = ax
            
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Value")
            ax.set_title(f"{mode.capitalize()} Mode")
            
            text_dict[mode] = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            lines_dict[mode] = {}
            visibility_dict[mode] = {}
            
            # Get all ports for this mode
            ports_in_mode = sorted(list(data_map[mode].keys()))
            
            # Create checkbox axes (positioned to the right of the plot, larger size)
            checkbox_ax = fig.add_axes([0.76, 0.1 + idx * (0.8 / len(modes_in_data)), 0.22, 0.7 / len(modes_in_data)])
            checkbox_axes_dict[mode] = checkbox_ax
            
            # Create lines and visibility tracking
            for port in ports_in_mode:
                lines_dict[mode][port] = ax.plot([], [], label=port, marker='', markersize=2, linewidth=1.5)[0]
                visibility_dict[mode][port] = True  # Start with all visible
            
            # Create checkboxes with larger spacing to accommodate bigger boxes
            checkbox_labels = ports_in_mode
            checkbox_states = [True] * len(checkbox_labels)  # All initially checked
            
            checkbox_dict[mode] = CheckButtons(checkbox_ax, checkbox_labels, checkbox_states)
            
            # FORCE larger checkbox rectangles - try multiple approaches
            try:
                # Method 1: Direct rectangle modification
                for i, rect in enumerate(checkbox_dict[mode].rectangles):
                    # Get current position
                    x, y = rect.get_xy()
                    # Set much larger dimensions
                    rect.set_width(0.08)  # Larger width in axes coordinates
                    rect.set_height(0.08)  # Larger height in axes coordinates
                    rect.set_xy((x, y))  # Ensure position is maintained
                    
                # Method 2: Also try setting linewidth to make border thicker
                for rect in checkbox_dict[mode].rectangles:
                    rect.set_linewidth(3)  # Thicker border
                    
            except AttributeError:
                print(f"Warning: Could not access rectangles directly for {mode}")
                
            # Try alternative approach - modify the axes properties
            try:
                # Set the checkbox axes to have larger tick marks or modify spacing
                checkbox_ax.tick_params(labelsize=16)
            except:
                pass
            
            # Make checkbox text larger
            try:
                for text in checkbox_dict[mode].labels:
                    text.set_fontsize(16)  # Even larger font
            except AttributeError:
                try:
                    for text in checkbox_dict[mode]._labels:
                        text.set_fontsize(16)  # Even larger font
                except AttributeError:
                    print(f"Warning: Could not resize checkbox text for {mode} mode")
            
            # Set up callback with mode context
            def make_callback(mode):
                return lambda label: on_checkbox_clicked(label, mode)
            
            checkbox_dict[mode].on_clicked(make_callback(mode))
            
            ax.legend(loc='upper right')
        
        plt.tight_layout()



# ==============================
# Main loop
# ==============================
# Declare global variable

print('starting')
try:
    while True:
        line = ser.readline().decode("utf-8").strip()
        if not line:
            continue

        parse_start = time.time()
        parsed = parse_line(line)
        if parsed is None:
            continue
        
        timestamp_delta, mode_str, port_str, value = parsed
        
        # Convert timestamp to absolute (assuming sensor sends absolute deci-milliseconds)
        last_absolute_timestamp += timestamp_delta
        
        #print(parsed, last_absolute_timestamp)
        # Store in global map: mode -> port -> {'timestamps': [], 'values': []}
        if mode_str not in sensor_data_map:
            sensor_data_map[mode_str] = {}
        if port_str not in sensor_data_map[mode_str]:
            sensor_data_map[mode_str][port_str] = {'timestamps': [], 'values': []}
        
        sensor_data_map[mode_str][port_str]['timestamps'].append(last_absolute_timestamp)
        sensor_data_map[mode_str][port_str]['values'].append(value)
        
        # Debug: print data rate every 100 readings
        if len(sensor_data_map[mode_str][port_str]['timestamps']) % 100 == 0:
            print(f"Mode {mode_str}, Port {port_str}: {len(sensor_data_map[mode_str][port_str]['timestamps'])} points, last timestamp: {last_absolute_timestamp:.3f}s")
        
        
        # Update plot structure after first 2 seconds, then every 2 seconds
        current_time = time.time()
        if current_time - last_plot_update >= plot_update_interval:
            cutoff_time = last_absolute_timestamp - max_time
            for mode in sensor_data_map:
                for port in sensor_data_map[mode]:
                    data = sensor_data_map[mode][port]
                    # Use list slicing instead of repeated pop(0) which is O(n) each time
                    if data['timestamps'] and data['timestamps'][0] < cutoff_time:
                        # Find the first index that should be kept
                        idx = 0
                        for i, ts in enumerate(data['timestamps']):
                            if ts >= cutoff_time:
                                idx = i
                                break
                        # Slice lists to remove old data efficiently
                        data['timestamps'] = data['timestamps'][idx:]
                        data['values'] = data['values'][idx:]

            data_map = post_process(sensor_data_map)
            update_plot_structure(data_map)
            last_plot_update = current_time
        
            # Get minimum timestamp (first non-empty list since data is sorted chronologically)
            min_timestamp = float('inf')
            for mode in data_map:
                for port in data_map[mode]:
                    data = data_map[mode][port]
                    if data['timestamps']:
                        min_timestamp = min(min_timestamp, data['timestamps'][0])
        
            # Update plot data dynamically
            for mode in data_map:
                if mode not in axes_dict:
                    continue
                
                ax = axes_dict[mode]

                # Update data for all ports
                for port in data_map[mode]:
                    if port in lines_dict.get(mode, {}):
                        data = data_map[mode][port]
                        line = lines_dict[mode][port]
                        
                        # Update data
                        if len(data['timestamps']) > 0:
                            line.set_data(data['timestamps'], data['values'])
                        else:
                            line.set_data([], [])
                        
                        # Apply visibility setting
                        if mode in visibility_dict and port in visibility_dict[mode]:
                            line.set_visible(visibility_dict[mode][port])

                # Trigger autoscaling for y-axis since data changed
                ax.relim(visible_only=True)
                ax.autoscale_view(tight=None, scalex = True, scaley=True)

                # # Update x-axis limits to show rolling window
                # ax.set_xlim(min_timestamp, last_absolute_timestamp)
        
            # Print number of x values for the first mode and port
            if data_map:
                first_mode = next(iter(data_map))
                if data_map[first_mode]:
                    first_port = next(iter(data_map[first_mode]))
                    num_x = len(data_map[first_mode][first_port]['timestamps'])
                    if first_mode in text_dict:
                        text_dict[first_mode].set_text(f"X values: {num_x}")
            # print(time.time()- current_time)
            
             # Check if matplotlib window is closed
            if not plt.fignum_exists(fig.number):
                print("\nMatplotlib window closed, stopping visualization")
                ser.close()
                break  
            # Update the plot display
            plt.pause(0.002)
 
except KeyboardInterrupt:
    print("\nStopped real-time visualization")
