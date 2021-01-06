"""Settings for audio reactive LED strip"""
from __future__ import print_function
from __future__ import division
import os

configurations = {
	'default': {
		# GUI Configuration
		# ----------------------------------
		# Whether or not to display a PyQtGraph GUI plot of visualization
		'USE_GUI': True,
		# Whether to display debug information
		'DEBUG': True,
		# Target GUI framerate. Will warn when this can't be met.
		'FPS_GUI': 60,

		# Alsa Configuration
		# ----------------------------------------------
		# Number of channels
		'CHANNELS': 2,
		# Input and output have to use the same sample-rate and number of channels.
		# Any unavoidable differences should be dealt with using alsa's pcm_rate.
		# Hardware sample rate
		'SAMPLE_RATE': 44100,
		# Alsa Device names (typically hw:0,0 or something like that)
		'ALSA_SOURCE': 'hw:3,1',
		'ALSA_SINK': 'pulse',

		# LED Output
		# ----------------------------------
		# IP address of the ESP8266. Must match IP in ws2812_controller.ino
		'UDP_IP': '192.168.178.175',
		# Port number used for socket communication between Python and ESP8266
		'UDP_PORT': 8080,
		# Number of pixels in the LED strip (must match ESP8266 firmware)
		'N_PIXELS': 300,
		# Target LED framerate. Will warn when this can't be met.
		'FPS_LED': 100,
		# Set to False because the firmware handles gamma correction + dither"""
		'SOFTWARE_GAMMA_CORRECTION': False,
		# Location of the gamma correction table
		'GAMMA_TABLE_PATH': os.path.join(os.path.dirname(__file__), 'gamma_table.npy'),

		# FFT Settings 
		# ----------------------------------
		# Frequencies below this value will be removed during audio processing
		'MIN_FREQUENCY': 20,
		# Frequencies above this value will be removed during audio processing
		'MAX_FREQUENCY': 20000,
		# Number of frequency bins to use when transforming audio to frequency domain
		'FFT_N_BINS': 150,
		#Length (ms) of the rolling audio window to be used. Will be adjusted to
		# improve fft performance.
		'FFT_WINDOW_LENGTH': 50
	}
}

# Return a dict of available visualizations
def visualizations(config):
	import numpy as np
	import dsp

	# A valid visualization is a function with the following signature:
	# Callable[[np.ndarray[float, FFT_N_BINS], np.ndarray[float, len(FFT_WINDOW)]]
	#  -> np.ndarray[int, 4, N_PIXELS]]

	def visualize_waveform(_, waveform):
	    interpolated = dsp.interpolate(waveform, config.N_PIXELS)
	    clipped = np.clip(interpolated - 0.5, 0, 1) * 50

	    zeros = np.zeros(config.N_PIXELS);
	    return np.array([zeros, zeros, zeros, clipped]);

	def visualize_spectrum(spectrum, _):
	    interpolated = dsp.interpolate(spectrum, config.N_PIXELS)
	    pixels = np.array([
	        # np.clip(1*np.log(interpolated*10), 0, 1),
	        # np.clip(0.3*np.log(interpolated*10), 0, 1),
	        # np.clip(0.3 * interpolated, 0, 1)
	        # np.tile(0, config.N_PIXELS),
	        # np.tile(0, config.N_PIXELS),
	        # np.clip(0.3 * interpolated, 0, 1),
	        interpolated,
	        interpolated,
	        interpolated,
	        interpolated
	    ])
	    return np.clip(pixels, 0, 1) * 255;

	return {
		'waveform': visualize_waveform,
		'spectrum': visualize_spectrum
	}
