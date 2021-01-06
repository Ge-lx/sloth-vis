from threading import Lock
import queue
import time

import alsaaudio
import numpy as np

from utils import setInterval, runAsync
from web import app
import config
import visualization
import gui
import led

fifo_visualize = queue.Queue(5)
fifo_output_sound = queue.Queue(8)

# fifo to block on for hardware-timed updates every period-length
fifo_output_vis = queue.Queue(1)
current_output_vis = None
current_output_vis_lock = Lock()

# Which output should be ratelocked to the visualization
visualization_ratelocked_led = config.FPS_LED > config.FPS_GUI

def reset_counters():
    global cnt_input, cnt_xruns_visualize, cnt_output_sound, cnt_visualize, \
           cnt_xruns_output, cnt_output_led, cnt_output_gui, time_a, time_b, time_c
    cnt_input = 0
    cnt_xruns_visualize = 0
    cnt_output_sound = 0

    cnt_visualize = 0
    cnt_xruns_output = 0

    cnt_output_led = 0
    cnt_output_gui = 0

    time_a = time_b = time_c = 0
reset_counters()

last_second = time.time()
def print_debug():
    global last_second, time_c, time_b
    if (time.time() - last_second > 1):
        last_second += 1

        time_total = time_c
        time_c -= time_b
        time_b -= time_a

        time_max_total = 1000/config.target_fft_fps

        print(f'# of periods     : IN {cnt_input:3.0F} -> OUT {cnt_output_sound:3.0F}  |  VIS  \
                {cnt_visualize:3.0F} ({cnt_xruns_visualize:1.0F})  |  OUT ({cnt_xruns_output:1.0F}) \
                - LED {cnt_output_led:3.0F} - GUI {cnt_output_gui:3.0F}')
        print(f'avg. frame-times : ROLL {time_a*1000:10.3F}ms  |  FFT {time_b*1000:6.3F}ms  |  VIS \
                {time_c*1000:9.3F}ms  -> {time_total*1000:.1F}/{time_max_total:.1F}ms\n')
        reset_counters()


def worker_input(alsa_source):
    global cnt_input, cnt_xruns_visualize
    while True:
        # Blocks on read when no input data available
        l, data = alsa_source.read()
        if l < 1: continue

        # Block on put to output_queue to time on output device clock
        fifo_output_sound.put(data)

        try:
            fifo_visualize.put(data, block=False)
        except queue.Full:
            cnt_xruns_visualize += 1          
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
    global cnt_visualize, cnt_xruns_output, current_output_vis, time_a, time_b, time_c
    while True:
        # Block on read when no data is available 
        frame = fifo_visualize.get()

        # Don't process if visualization is disabled
        if (state.visualization_enabled() == False):
            continue

        # Audio processing
        array_stereo = np.frombuffer(frame, dtype=np.dtype('<i2'))
        array_mono = array_stereo[::2]/2 + array_stereo[1::2]/2
        output, (time_a, time_b, time_c) = visualization.process_sample(array_mono)

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
        return

    global cnt_output_led
    if (visualization_ratelocked_led):
        # Block on read
        output = fifo_output_vis.get()
        led.update(output)
        fifo_output_vis.task_done()
    else:
        with current_output_vis_lock:
            output = current_output_vis

        if (output is not None):
            led.update(output)
    cnt_output_led += 1

def once_output_gui():
    if (state.visualization_enabled() == False):
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
    app.run()
    print(f'Started web sever.')

    # Start sound workers
    alsa_source = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,
        format = alsaaudio.PCM_FORMAT_S16_LE,
        channels = config.CHANNELS,
        rate = config.SAMPLE_RATE,
        device = config.ALSA_SOURCE,
        periodsize = config.fft_samples_per_update)

    alsa_sink = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK,
        format = alsaaudio.PCM_FORMAT_S16_LE,
        channels = config.CHANNELS,
        rate = config.SAMPLE_RATE,
        device = config.ALSA_SINK,
        periodsize = config.fft_samples_per_update)

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
        setInterval(once_output_gui, 1/config.FPS_GUI)
    else:
        def worker_output_gui():
            while True:
                once_output_gui()

        runAsync(worker_output_gui)
        setInterval(once_output_led, 1/config.FPS_LED)

    
    print(f'Threads started.\n\n')
    if config.USE_GUI:
        gui.init(tick_callback=print_debug if config.DEBUG else lambda: None)
    else:
        setInterval(print_debug, 1)
