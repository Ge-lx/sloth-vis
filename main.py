from threading import Lock
import queue
import time

import alsaaudio
import numpy as np

from utils import setInterval, runAsync
# from web import app
import control_socket
import state
import visualization
import gui
import led

config = state.default_config

fifo_visualize = queue.Queue(4)
fifo_output_sound = queue.Queue(7)

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

    def log(length):
        nonlocal last, start, measurements
        if (len(measurements.keys()) == 0):
            return

        total_time = sum([sum(times) for _, times in measurements.items()])
        counts = len(measurements[list(measurements.keys())[0]])
        # mean_frame_time = total_time / counts

        results = sorted([({
            'name': name, \
            'mean': np.mean(diff_times), \
            'share': np.sum(diff_times) / total_time \
        }) for name, diff_times in measurements.items()], key=lambda x: x['name'])

        # results.sort(lambda x: x['name'])

        out_names = '|'
        out_times = '|'
        space_loan = 0

        def generate_strings(e):
            nonlocal out_names, out_times, space_loan
            # l = int(length * e['share'] + 0.5) - space_loan

            out_names += f'{e["name"]} : {e["mean"] * 1000:.1F}ms'.center(20) + '|'

            # time = f'{e["mean"] * 1000:.1F}ms'.center(l-1) + '|'
            # name = f'{e["name"]}'.center(len(time)-1) + '|'

            # out_names += name
            # out_times += time

            # final_length = max(len(name), len(time))
            # if (final_length > l):
            #     space_loan = final_length - l
            # else:
            #     space_loan = 0

        [generate_strings(e) for e in results]

        # assert(len(out_names) == length)
        # assert(len(out_times) == length)

        print(out_names)
        # print(out_times)


    return {'reset': reset, 'measure': measure, 'log': log}

logger = performance_logger()

def reset_counters():
    global cnt_input, cnt_xruns_visualize, cnt_output_sound, cnt_visualize, \
           cnt_xruns_output, cnt_output_led, cnt_output_gui
    cnt_input = 0
    cnt_xruns_visualize = 0
    cnt_output_sound = 0

    cnt_visualize = 0
    cnt_xruns_output = 0

    cnt_output_led = 0
    cnt_output_gui = 0
reset_counters()

last_second = time.time()
def print_debug():
    global last_second
    if (time.time() - last_second > 1):
        last_second += 1

        if (config["DEBUG"]):
            logger['log'](100)
            logger['reset']()
            print(f'Queue throughput  : IN {cnt_input:3.0F} -> OUT {cnt_output_sound:3.0F}  |  VIS  {cnt_visualize:3.0F} ({cnt_xruns_visualize:1.0F})  |  OUT ({cnt_xruns_output:1.0F}) - LED {cnt_output_led:3.0F} - GUI {cnt_output_gui:3.0F}\n')
        reset_counters()



def worker_input(alsa_source):
    global cnt_input, cnt_xruns_visualize
    while True:
        # Blocks on read when no input data available
        l, data = alsa_source.read()
        if l < 1: continue

        if (state.visualization_enabled()):
            try:
                fifo_visualize.put(data, block=False)
            except queue.Full:
                cnt_xruns_visualize += 1

        # Block on put to output_queue to time on output device clock
        fifo_output_sound.put(data)

        cnt_input += 1


def worker_output_sound(alsa_sink):
    global cnt_output_sound
    while not fifo_output_sound.full():
        time.sleep(0.1)

    while True:
        frame = fifo_output_sound.get()

        # Block on audio output to use hardware timer
        alsa_sink.write(frame)
        fifo_output_sound.task_done()
        cnt_output_sound += 1

def worker_visualize():
    global cnt_visualize, cnt_xruns_output, current_output_vis
    while True:
        # Block on read when no data is available 
        frame = fifo_visualize.get()

        # # Don't process if visualization is disabled
        #  == False):
        #     continue

        # Audio processing
        array_stereo = np.frombuffer(frame, dtype=np.dtype('<i2'))
        array_mono = array_stereo[::2]/2 + array_stereo[1::2]/2

        res = visualization.process_sample(array_mono, logger['measure'])
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
    # Start sound workers
    alsa_source = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,
        format = alsaaudio.PCM_FORMAT_S16_LE,
        channels = config["CHANNELS"],
        rate = config["SAMPLE_RATE"],
        device = config["ALSA_SOURCE"],
        periodsize = config["fft_samples_per_update"])

    alsa_sink = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK,
        format = alsaaudio.PCM_FORMAT_S16_LE,
        channels = config["CHANNELS"],
        rate = config["SAMPLE_RATE"],
        device = config["ALSA_SINK"],
        periodsize = config["fft_samples_per_update"])

    runAsync(lambda: worker_output_sound(alsa_sink))
    runAsync(lambda: worker_input(alsa_source))

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

    if config["USE_GUI"]:
        gui.init(config=config, tick_callback=print_debug if config["DEBUG"] else lambda: None)
    else:
        setInterval(print_debug, 1)
        control_socket.start_control_socket()
        # app.run(host='0.0.0.0')
