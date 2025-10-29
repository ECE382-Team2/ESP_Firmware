import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import serial
# import joblib
# ==============================
# predicition fucntion
# ==============================

def calculate_ft_from_psoc(c1n, c2n, c3n, c4n, c1s, c2s, c3s, c4s):
    """
    Calculate force/torque from 8 PSOC sensor inputs using 2nd-order polynomial regression.
    
    Args:
        c1n, c2n, c3n, c4n: Normal sensors 1-4
        c1s, c2s, c3s, c4s: Shear sensors 1-4
    
    Returns:
        np.array: [fx, fy, fz, tx, ty, tz]
    """
    # Model intercepts
    fx_intercept = -6.018745857180133e+04
    fy_intercept = -2.016358749653060e+04
    fz_intercept = 4.119084927626266e+05
    tx_intercept = -8.804094328781720e+02
    ty_intercept = -3.610360020036960e+03
    tz_intercept = -1.284116167795461e+03
    
    # Model coefficients (45 polynomial features each)
    fx_coef = np.array([
        3.046183461811196e-07, 3.915173383998768e-01, 4.919275893827008e-01, 1.814119519290021e-01, -7.229482869826916e-03,
        -1.815187667211892e+00, 2.737244577593229e-01, -5.299132964183279e-01, 7.292742457467025e-01, -9.316079116213718e-07,
        -1.193962417393207e-06, 2.899563027464515e-07, -7.104908265670494e-07, -4.125882053836821e-06, 8.630924856239760e-06,
        -8.098849493079892e-07, -2.726979417031757e-07, -9.651046054970302e-07, -1.972049379164448e-06, 8.353310119090770e-07,
        2.061693232475170e-05, -8.520498953079890e-06, 6.026333615585179e-06, -6.264458969068996e-06, 3.357637251991820e-08,
        4.841650995539879e-09, 1.811811198130478e-06, -1.651376821184664e-06, -8.363604970461565e-08, -2.038558066286701e-07,
        -3.605777446928149e-08, -2.547238739457244e-06, -3.996451030635204e-07, -3.246504029687930e-07, 2.085980041442725e-07,
        -8.449646923320912e-05, 1.523224007034791e-04, -2.466142579634608e-06, 3.841388038828940e-06, -5.336576390777548e-05,
        2.556574141098938e-05, -1.559162681015491e-05, -1.472358612444457e-07, -4.181690015372049e-06, 2.464001289340253e-07
    ])
    
    fy_coef = np.array([
        2.305747385103482e-06, 1.953131646367295e-02, 2.351761635323825e-01, 7.055264424244227e-02, -2.410359779319578e-02,
        -1.290470723448289e+01, 1.831255948179031e+00, 3.597199040086411e-01, 2.104849843515600e-02, 1.789072397937231e-06,
        -3.351424380391913e-06, -2.642811608213860e-07, 4.307992367794790e-09, 4.720446561064268e-05, -2.314063021280515e-05,
        3.854105250577850e-07, -6.638678520281561e-08, 9.074885014118278e-07, -2.724612216118482e-07, 1.330267822029424e-07,
        7.942752050579357e-05, 6.214106582463861e-06, -3.847529715772655e-06, 5.445278490060610e-08, -4.984217116595596e-08,
        5.696535050084853e-09, -6.485446140649453e-07, -2.908226671568743e-06, 3.327869933189137e-08, 7.046780960429125e-08,
        2.707795059284621e-08, -7.245507395893706e-06, 6.691026335555519e-06, 2.551868796451173e-07, -1.458136687667016e-07,
        4.143664070974546e-04, 2.217680635370080e-04, 7.912205019038958e-07, 1.285855254785846e-05, 4.006875565684273e-04,
        1.607398805464470e-05, 1.112492860243909e-05, 2.322774736458535e-07, 2.838201319897904e-06, 3.061783690315489e-07
    ])
    
    fz_coef = np.array([
        1.891128009663307e-05, -7.398897508170579e-01, -8.370138753089616e+00, 7.612665051483160e-01, 3.384519456483315e-01,
        -1.158865106260597e+02, -2.076698839480769e+01, 4.635287877579364e+00, 1.146128747175233e+00, 2.667024544031708e-05,
        -4.232881988709222e-05, 5.093109722689916e-07, -4.370616155286890e-06, 3.893508421053627e-04, -4.035757564230536e-04,
        -1.359991746780754e-06, 3.844288954687172e-06, 6.293251549453809e-05, -5.757012885508842e-06, 2.126815445737850e-06,
        6.872152567925470e-04, 5.786121158844332e-04, -4.314453732077377e-05, -1.341498491952303e-05, -6.708039686777743e-07,
        1.936785130136236e-09, 1.232303359146689e-05, -2.607411341832136e-06, 8.854833393550917e-07, -2.506179905827060e-06,
        -3.845060265021305e-07, -4.222097488303497e-05, 4.190176365140747e-05, 2.726376892183899e-06, 2.054995073117252e-06,
        2.149951035012723e-03, -1.392477546453137e-04, 6.429858751689370e-05, 3.991665376572134e-05, 5.554807207420643e-03,
        8.531747539336935e-05, 5.005584877873215e-05, 3.321188835297690e-06, -1.005078506374844e-06, -1.029682398077816e-06
    ])
    
    tx_coef = np.array([
        1.498639606169316e-07, -4.710875444588795e-02, 4.960105440245996e-02, 3.114757774792658e-03, 4.906874153724602e-03,
        -9.123405020614327e-01, -1.046435418353015e-01, 7.507823714662149e-03, 2.557828177118919e-02, 1.653356987237570e-07,
        1.511084705400859e-07, 8.542632098401105e-11, -5.178641984211277e-08, 2.341936626689094e-06, 6.281653872869586e-07,
        -2.490999532565620e-08, -4.604481476278166e-08, -2.658905373188212e-07, -2.775735156993019e-08, 4.037218061973281e-09,
        6.258669087490402e-06, 3.554908696040742e-07, -4.817775551612330e-08, -1.789304653498495e-07, -7.260202999878166e-12,
        -4.576213941790639e-12, -3.953057348819909e-08, 5.950300794767322e-08, 6.924015349954611e-09, 7.643916437059503e-09,
        1.365979193921765e-09, -2.496769964580067e-07, -1.331575497528635e-07, 3.789912036544080e-10, -8.530773604772772e-09,
        1.859731144116051e-05, 1.765428693566919e-05, -1.269884947950147e-07, 9.113083562144250e-07, -1.626973316533636e-05,
        6.002256323037710e-07, -6.657513344545699e-08, -6.613249751564034e-09, -1.996142838061263e-08, 7.870376024096887e-08
    ])
    
    ty_coef = np.array([
        -3.054778366279614e-08, 4.080036379369498e-03, 6.455896773566308e-02, -2.519378718476815e-03, 5.213285370174100e-04,
        2.999187441123031e-01, 4.016832552392534e-01, -2.914926456904444e-02, -2.987518490389711e-02, -5.740116449140549e-08,
        5.304509247214480e-08, 3.027554925698434e-09, 1.789495512020792e-08, -7.503287142635383e-07, -2.510296624663333e-07,
        -3.601986388677124e-08, -1.738841893467529e-08, -3.186965314261205e-07, 1.233538764085252e-08, -2.101609469150504e-08,
        -2.258399732321251e-06, -3.659915082183491e-06, 2.952074074910654e-07, 2.749275716515098e-07, 3.717719337133480e-09,
        -4.775597276627664e-10, 2.428458525008034e-08, 1.139677792071488e-07, -2.313360619990048e-09, -1.365474823361054e-08,
        -1.844826073537721e-11, 1.803718055771793e-07, 9.504777107236968e-08, -4.282003749625160e-09, 2.309002059161741e-08,
        -1.116614856523486e-05, -1.524053529297285e-05, -1.382577649195799e-06, -5.535077068238972e-07, -1.112444861876622e-05,
        -5.092935812181505e-07, 1.166935918947252e-06, -8.222909739007627e-09, -1.667156771427134e-07, -5.498869872222876e-08
    ])
    
    tz_coef = np.array([
        3.782240326022936e-09, 1.023246771070913e-02, 1.174522024606123e-02, 2.776361865601717e-03, -2.239223780277097e-03,
        -3.233282603388142e-02, -2.151158198875897e-02, 3.964280497141005e-03, 1.399370052095758e-02, -1.360586430086911e-08,
        -5.666740246673933e-08, -3.634753168883648e-09, -1.926432889335388e-09, 2.208589143593105e-07, 1.322281520912465e-07,
        -1.703753317445633e-08, -1.002197751552945e-08, -2.258238229456140e-08, -2.095371516628655e-08, 2.029986876745389e-08,
        1.598311972375765e-07, 1.223260345829473e-07, -1.165313089531985e-08, -1.122350299006446e-07, -3.741056061869217e-10,
        5.811456577373002e-10, -3.084835668211150e-08, -1.788961469170995e-08, -1.256666332523641e-09, -4.354701869684359e-09,
        2.910469456058472e-10, -4.845806378025476e-08, -2.880062185697404e-09, -2.769837564902322e-09, 1.227836197217067e-09,
        1.880734618659632e-06, -7.866807046982829e-07, -3.355827279473546e-08, -5.830128797156929e-09, 6.935305231410458e-06,
        6.284663464574027e-07, -4.714055740438685e-07, -4.553248902169446e-09, -7.586507373878195e-08, -1.831791662572408e-09
    ])
    
    # Create input array
    input_arr = np.array([c1n, c2n, c3n, c4n, c1s, c2s, c3s, c4s])
    
    # Calculate polynomial features (degree 2)
    poly_features = [1.0]  # Constant term
    
    # Linear terms (8 features)
    poly_features.extend(input_arr)
    
    # Quadratic and interaction terms (36 features)
    for i in range(8):
        for j in range(i, 8):
            poly_features.append(input_arr[i] * input_arr[j])
    
    poly_features = np.array(poly_features)
    
    # Calculate predictions
    fx = fx_intercept + np.dot(fx_coef, poly_features)
    fy = fy_intercept + np.dot(fy_coef, poly_features)
    fz = fz_intercept + np.dot(fz_coef, poly_features)
    tx = tx_intercept + np.dot(tx_coef, poly_features)
    ty = ty_intercept + np.dot(ty_coef, poly_features)
    tz = tz_intercept + np.dot(tz_coef, poly_features)
    
    return np.array([fx, fy, fz, tx, ty, tz])

# ==============================
# Derivative shit that hopes the sensativity of the sensor is consistent
# ==============================
# Previous capacitance values for derivative calculation
global sensor_data_map


SERIAL_PORT = '/dev/ttyACM0'  # serial port to connect to

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

def update_plot_structure():
    """Dynamically create/update plot structure based on current modes in sensor_data_map"""
    global fig, axes_dict, lines_dict, text_dict, checkbox_dict, visibility_dict, checkbox_axes_dict
    
    modes_in_data = list(sensor_data_map.keys())
    
    # If no modes yet, skip
    if len(modes_in_data) == 0:
        return
    
    # Check if we need to recreate the figure
    modes_changed = set(modes_in_data) != set(axes_dict.keys())
    
    # Check if ports have changed for existing modes
    ports_changed = False
    for mode in modes_in_data:
        if mode in axes_dict:
            current_ports = set(sensor_data_map[mode].keys())
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
            ports_in_mode = sorted(list(sensor_data_map[mode].keys()))
            
            # Create checkbox axes (positioned to the right of the plot, larger size)
            checkbox_ax = fig.add_axes([0.76, 0.1 + idx * (0.8 / len(modes_in_data)), 0.22, 0.7 / len(modes_in_data)])
            checkbox_axes_dict[mode] = checkbox_ax
            
            # Create lines and visibility tracking
            for port in ports_in_mode:
                lines_dict[mode][port] = ax.plot([], [], label=port, marker='', markersize=2, linewidth=1.5)[0]
                visibility_dict[mode][port] = True  # Start with all visible
            
            # Create checkboxes
            checkbox_labels = ports_in_mode
            checkbox_states = [True] * len(checkbox_labels)  # All initially checked
            
            checkbox_dict[mode] = CheckButtons(checkbox_ax, checkbox_labels, checkbox_states)
            
            # Make checkboxes bigger
            for rect in checkbox_dict[mode].rectangles:
                rect.set_width(0.15)  # Increase checkbox width
                rect.set_height(0.08)  # Increase checkbox height
            
            # Make checkbox text larger
            for text in checkbox_dict[mode].labels:
                text.set_fontsize(12)
            
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

            update_plot_structure()
            last_plot_update = current_time
        
            # Get minimum timestamp (first non-empty list since data is sorted chronologically)
            min_timestamp = float('inf')
            for mode in sensor_data_map:
                for port in sensor_data_map[mode]:
                    data = sensor_data_map[mode][port]
                    if data['timestamps']:
                        min_timestamp = min(min_timestamp, data['timestamps'][0])
        
            # Update plot data dynamically
            for mode in sensor_data_map:
                if mode not in axes_dict:
                    continue
                
                ax = axes_dict[mode]

                # Update data for all ports
                for port in sensor_data_map[mode]:
                    if port in lines_dict.get(mode, {}):
                        data = sensor_data_map[mode][port]
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
                ax.relim()
                ax.autoscale_view(scaley=True)

                # Update x-axis limits to show rolling window
                ax.set_xlim(min_timestamp, last_absolute_timestamp)
        
            # Print number of x values for the first mode and port
            if sensor_data_map:
                first_mode = next(iter(sensor_data_map))
                if sensor_data_map[first_mode]:
                    first_port = next(iter(sensor_data_map[first_mode]))
                    num_x = len(sensor_data_map[first_mode][first_port]['timestamps'])
                    if first_mode in text_dict:
                        text_dict[first_mode].set_text(f"X values: {num_x}")
            # print(time.time()- current_time)
            
             # Check if matplotlib window is closed
            if not plt.fignum_exists(fig.number):
                #print(sensor_data_map)
                print("\nMatplotlib window closed, stopping visualization")
                ser.close()
                break  
            # Update the plot display
            plt.pause(0.002)
 
except KeyboardInterrupt:
    print("\nStopped real-time visualization")
