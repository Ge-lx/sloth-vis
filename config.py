"""Settings for audio reactive LED strip"""
from __future__ import print_function
from __future__ import division
import os
import numpy as np

# GUI Configuration
# ----------------------------------

# Whether or not to display a PyQtGraph GUI plot of visualization
USE_GUI = True
# Whether to display debug information
DEBUG = True

# Target GUI framerate. Will warn when this can't be met.
FPS_GUI = 60


# Alsa Configuration
# ----------------------------------------------

# Input and output have to use the same sample-rate and number of channels.
# Any unavoidable differences should be dealt with using alsa's pcm_rate.

# Number of channels
CHANNELS = 2

# Hardware sample rate
SAMPLE_RATE = 44100

# Alsa Device names (typically hw:0,0 or something like that)
ALSA_SOURCE = 'hw:3,1'
ALSA_SINK = 'pulse'


# LED Output
# ----------------------------------

# IP address of the ESP8266. Must match IP in ws2812_controller.ino
UDP_IP = '192.168.178.175'
# Port number used for socket communication between Python and ESP8266
UDP_PORT = 8080

# Number of pixels in the LED strip (must match ESP8266 firmware)
N_PIXELS = 300

# Target LED framerate. Will warn when this can't be met.
FPS_LED = 100

# Set to False because the firmware handles gamma correction + dither"""
SOFTWARE_GAMMA_CORRECTION = False

# Location of the gamma correction table
GAMMA_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'gamma_table.npy')


# FFT Settings 
# ----------------------------------

# Frequencies below this value will be removed during audio processing
MIN_FREQUENCY = 20
# Frequencies above this value will be removed during audio processing
MAX_FREQUENCY = 20000
# Number of frequency bins to use when transforming audio to frequency domain
FFT_N_BINS = 150

# Length (ms) of the rolling audio window to be used. Will be adjusted to
# improve fft performance.
FFT_WINDOW_LENGTH = 50


# Validation and tuning
# ----------------------------------

fft_samples_per_window = (FFT_WINDOW_LENGTH / 1000) * SAMPLE_RATE
target_fft_fps = max(FPS_LED, FPS_GUI);
fft_samples_per_update = int(SAMPLE_RATE / target_fft_fps)

if (DEBUG):
    print(f'Sampling audio at {SAMPLE_RATE} Hz.')
    print(f'Target window of {FFT_WINDOW_LENGTH} ms -> {fft_samples_per_window} samples per window.')
    print(f'Theoretical minimum frequency of {(1 / (FFT_WINDOW_LENGTH / 1000)):.1F} Hz.')
    faster_output = 'GUI' if FPS_GUI > FPS_LED else 'LED'
    print(f'Target fps of {target_fft_fps} Hz for {faster_output}.')

fft_samples_per_window = 2**int(np.ceil(np.log2(fft_samples_per_window)))

if (DEBUG):
    fft_window_secs = fft_samples_per_window / SAMPLE_RATE
    print(f'Increasing fft window to {fft_samples_per_window} samples -> {(fft_window_secs * 1000):.1F} ms, {(1 / fft_window_secs):.1F} Hz')
    print(f'Using audio period size of {fft_samples_per_update} samples.')
