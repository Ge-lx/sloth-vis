from threading import Lock
import queue
import time

import audio_backend.backend_pulseaudio as pulseaudio
import numpy as np

from utils import setInterval, runAsync, setTimeout
# from web import app
import control_socket
import state
import visualization
import gui
import led

config = state.default_config

fifo_visualize = queue.Queue(1)

# fifo to block on for hardware-timed updates every period-length
fifo_output_vis = queue.Queue(1)
current_output_vis = None
current_output_vis_lock = Lock()

# Which output should be ratelocked to the visualization
visualization_ratelocked_led = config["FPS_LED"] > config["FPS_GUI"]

def performance_logger():
    measurements = dict()
    start = time.time()
    last = start

    def reset():
        nonlocal last, start, measurements
        measurements = dict()
        start = time.time()
        last = start

    def measure(name):
        nonlocal last, start, measurements
        now = time.time()
        diff = now - last
        last = now

        if name in measurements:
            measurements[name] += [diff]
        else:
            measurements[name] = [diff]

    def log(prefix = ''):
        nonlocal last, start, measurements
        if (len(measurements.keys()) == 0):
            return

        total_time = sum([sum(times) for _, times in measurements.items()])
        counts = len(measurements[list(measurements.keys())[0]])

        results = sorted([({
            'name': name, \
            'mean': np.mean(diff_times), \
            'share': np.sum(diff_times) / total_time \
        }) for name, diff_times in measurements.items()], key=lambda x: x['name'])

        out_string = '|'

        def generate_strings(e):
            nonlocal out_string
            out_string += f'{e["name"]} {e["mean"] * 1000:.1F}ms'.center(20) + '|'
        [generate_strings(e) for e in results]

        print(f'{prefix} {out_string}')

    return {'reset': reset, 'measure': measure, 'log': log}

logger = performance_logger()

def reset_counters():
    global cnt_input, cnt_xruns_visualize, cnt_visualize, \
           cnt_xruns_output, cnt_output_led, cnt_output_gui
    cnt_input = 0
    cnt_xruns_visualize = 0

    cnt_visualize = 0
    cnt_xruns_output = 0

    cnt_output_led = 0
    cnt_output_gui = 0
reset_counters()

last_second = time.time()
latency = 0
fragment_size = 0
def print_debug():
    global last_second
    if (time.time() - last_second > 1):
        last_second += 1

        if (config["DEBUG"]):
            print(f'----------------------------------------------------------------------------------------------------------|')
            print(f'Audio timings         |  latency {latency:3.2F}ms  |  fragsize  {fragment_size:5.0F}   |                    |                    |')
            logger['log']('Visualization timings')
            logger['reset']()
            print(f'Queue throughput      |       IN {cnt_input:3.0F}       |    VIS  {cnt_visualize:3.0F} ({cnt_xruns_visualize:1.0F})    |     OUT ({cnt_xruns_output:1.0F}) - LED {cnt_output_led:3.0F} - GUI {cnt_output_gui:3.0F}         |')

        reset_counters()


def process_source_buffer (l, data, idx):
    global cnt_input, cnt_xruns_visualize

    if (state.visualization_enabled()):
        try:
            frame = np.frombuffer(data, dtype=np.dtype('<i2')).copy()
            fifo_visualize.put((frame, idx), block=True)
        except queue.Full:
            cnt_xruns_visualize += 1

    cnt_input += 1

def worker_visualize():
    global cnt_visualize, cnt_xruns_output, current_output_vis

    while True:
        # Block on read when no data is available 
        while not fifo_visualize.full():
            time.sleep(0.001)

        (array_stereo, idx) = fifo_visualize.get()

        # Don't process if visualization is disabled
        if (state.visualization_enabled() == False):
            continue

        # Audio processing
        array_mono = array_stereo[::2]/2 + array_stereo[1::2]/2

        res = visualization.process_sample(array_mono, idx, logger['measure'])
        if (res == None):
            continue
        output = res

        try:
            fifo_output_vis.put(output, block=False)
        except queue.Full:
            cnt_xruns_output += 1

        with current_output_vis_lock:
            current_output_vis = output

        fifo_visualize.task_done()
        cnt_visualize += 1

def once_output_led():
    if (state.visualization_enabled() == False):
        time.sleep(0.5)
        return

    global cnt_output_led
    if (visualization_ratelocked_led):
        # Block on read
        output = fifo_output_vis.get()
        if (state.visualization_enabled()):
            led.update(output)
        fifo_output_vis.task_done()
    else:
        with current_output_vis_lock:
            output = current_output_vis

        if (output is not None and state.visualization_enabled()):
            led.update(output)
    cnt_output_led += 1

def once_output_gui():
    if (state.visualization_enabled() == False):
        time.sleep(0.5)
        return

    global cnt_output_gui
    if (visualization_ratelocked_led):
        with current_output_vis_lock:
            output = current_output_vis

        if (output is not None):
            gui.update(output)
    else:
        # Block on read
        output = fifo_output_vis.get()
        gui.update(output)
        fifo_output_vis.task_done()
    cnt_output_gui += 1

if __name__ == '__main__':


    def timing_update_handler(timing_info):
        global latency, fragment_size
        buffer_attributes = timing_info['buffer_attributes']
        fragment_size = buffer_attributes['fragsize']
        latency = timing_info['stream_latency'] / 1000

        # general = timing_info['general']
        # print(f'Timing:     Latency: {stream_latency / 1000:.2F} ms  |  Fragment size: {buffer_attributes["fragsize"]}')
        # print(general)
        # print(f'            Source latency: {general[''] / 1000:.2F} ms  |  Fragment size: {buffer_attributes["fragsize"]}')

    # Start pulseaudio backend
    pulseaudio.start_backend(config, process_source_buffer, timing_update_handler)
    
    # Start visualization worker
    runAsync(worker_visualize)

    # Start output workers
    if (visualization_ratelocked_led):
        def worker_output_led():
            while True:
                once_output_led()

        runAsync(worker_output_led)
        setInterval(once_output_gui, 1/config["FPS_GUI"])
    else:
        def worker_output_gui():
            while True:
                once_output_gui()

        runAsync(worker_output_gui)
        setInterval(once_output_led, 1/config["FPS_LED"])


    print(f'Threads started.\n\n')

    setTimeout(lambda: state.enable_visualization(), 1)

    if config["USE_GUI"]:
        gui.init(cfg=config, tick_callback=print_debug if config["DEBUG"] else lambda: None)
    else:
        setInterval(print_debug, 1)
        control_socket.start_control_socket()
        # app.run(host='0.0.0.0')
