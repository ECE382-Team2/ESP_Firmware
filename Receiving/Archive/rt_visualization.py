import time
import numpy as np
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports

NUM_SENSORS = 4
max_points = 100  # how many points to keep in history (window length)

# ==============================
# Setup serial connection (or FakeSerial for testing)
# ==============================
class FakeSerial:
    def __init__(self, mode=None):
        # mode: None -> alternate lines with a leading mode flag (old behavior)
        #       0 or 1 -> always produce values only for that port (no leading mode)
        self.counter = 0
        self.fixed_mode = mode

    def readline(self):
        values = np.random.randint(1, 100, NUM_SENSORS)
        time.sleep(0.02)
        if self.fixed_mode is None:
            mode = self.counter % 2
            self.counter += 1
            return (f"{mode}," + ",".join(map(str, values)) + "\n").encode("utf-8")
        else:
            return (",".join(map(str, values)) + "\n").encode("utf-8")

def open_serial(preferred_port="COM6", baud=9600, timeout=0.01, fake_mode=None):
    try:
        ser = serial.Serial(preferred_port, baud, timeout=timeout)
        print(f"Connected to {preferred_port}")
        return ser
    except Exception:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        for p in ports:
            try:
                ser = serial.Serial(p, baud, timeout=timeout)
                print(f"Connected to {p}")
                return ser
            except Exception:
                continue
    print("Using FakeSerial() for testing (mode={})".format(fake_mode))
    return FakeSerial(mode=fake_mode)

# Open a single serial on COM6
ser = open_serial(preferred_port="COM6", baud=9600, timeout=0.01, fake_mode=None)

# ==============================
# Setup plots: separate figs for normal and shear
# ==============================
plt.ion()
fig_norm, ax_norm = plt.subplots()
fig_shear, ax_shear = plt.subplots()

for ax in (ax_norm, ax_shear):
    ax.set_xlim(0, max_points)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Capacitance")
    ax.grid(True)

plot_keys = [f"C{i+1}" for i in range(NUM_SENSORS)]

lines_norm = {}
lines_shear = {}
for key in plot_keys:
    lines_norm[key] = ax_norm.plot([], [], label=key)[0]
    lines_shear[key] = ax_shear.plot([], [], label=key)[0]

ax_norm.legend(loc="upper right")
ax_shear.legend(loc="upper right")
fig_norm.suptitle("Normal Data")
fig_shear.suptitle("Shear Data")

history_norm = {key: [] for key in plot_keys}
history_shear = {key: [] for key in plot_keys}

# Helper to parse a line robustly:
def parse_line_to_values(line):
    # returns (mode_hint, values) where mode_hint is None, 0, or 1
    parts = [p.strip() for p in line.split(",") if p.strip() != ""]
    if len(parts) >= NUM_SENSORS + 1:
        try:
            mode_flag = int(parts[0])
            mode_flag = 0 if mode_flag == 0 else 1
            values = list(map(float, parts[1:1 + NUM_SENSORS]))
            return mode_flag, values
        except Exception:
            pass
    if len(parts) >= NUM_SENSORS:
        try:
            values = list(map(float, parts[0:NUM_SENSORS]))
            return None, values
        except Exception:
            pass
    return None, None

def autoscale_axis_from_history(ax, histories, current_len):
    # histories: dict of lists
    if current_len == 0:
        return
    vals = []
    for lst in histories.values():
        vals.extend(lst)
    if not vals:
        return
    y_min = min(vals)
    y_max = max(vals)
    # ensure a non-zero pad
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 1.0)
    else:
        pad = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(0, max(current_len, 1))

# ==============================
# Main loop: read single serial and route by leading flag
# ==============================
try:
    while True:
        updated_norm = False
        updated_shear = False

        # read from single port
        try:
            raw = ser.readline()
        except Exception:
            raw = b""
        if raw:
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                line = ""
            if line:
                print("RECV:", line, flush=True)
                mode_hint, values = parse_line_to_values(line)
                if values is not None:
                    # If mode_hint == 1 -> shear, mode_hint == 0 -> normal
                    # If no mode hint (None), treat as normal by default
                    if mode_hint == 1:
                        for i, key in enumerate(plot_keys):
                            history_shear[key].append(values[i])
                            if len(history_shear[key]) > max_points:
                                history_shear[key].pop(0)
                        updated_shear = True
                    else:
                        for i, key in enumerate(plot_keys):
                            history_norm[key].append(values[i])
                            if len(history_norm[key]) > max_points:
                                history_norm[key].pop(0)
                        updated_norm = True

        # Update normal plot if changed
        if updated_norm:
            current_len = len(history_norm[plot_keys[0]])
            x = list(range(current_len))
            for key in plot_keys:
                lines_norm[key].set_data(x, history_norm[key])
            autoscale_axis_from_history(ax_norm, history_norm, current_len)
            fig_norm.canvas.draw()
            fig_norm.canvas.flush_events()

        # Update shear plot if changed
        if updated_shear:
            current_len = len(history_shear[plot_keys[0]])
            x = list(range(current_len))
            for key in plot_keys:
                lines_shear[key].set_data(x, history_shear[key])
            autoscale_axis_from_history(ax_shear, history_shear, current_len)
            fig_shear.canvas.draw()
            fig_shear.canvas.flush_events()

        # small sleep to avoid 100% CPU
        time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopped real-time visualization")
