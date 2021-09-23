"""Settings for audio reactive LED strip"""
from __future__ import print_function
from __future__ import division
import os
from utils import hsv2rgb
import dsp

configurations = {
	'default': {
		# GUI Configuration
		# ----------------------------------
		# Whether or not to display a PyQtGraph GUI plot of visualization
		'USE_GUI': False,
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
		'ALSA_SOURCE': 'hw:3,0',#'pulse',
		'ALSA_SINK': 'pulse',#'rate_convert',

		# LED Output
		# ----------------------------------
		# IP address of the ESP8266. Must match IP in ws2812_controller.ino
		'UDP_IP': '192.168.43.21',
		# Port number used for socket communication between Python and ESP8266
		'UDP_PORT': 8080,
		# Number of pixels in the LED strip (must match ESP8266 firmware)
		'N_PIXELS': 40,
		# Target LED framerate. Will warn when this can't be met.
		'FPS_LED': 40,
		# Set to False because the firmware handles gamma correction + dither"""
		'SOFTWARE_GAMMA_CORRECTION': False,
		# Location of the gamma correction table
		'GAMMA_TABLE_PATH': os.path.join(os.path.dirname(__file__), 'gamma_table.npy'),

		# FFT Settings 
		# ----------------------------------
		# Frequencies below this value will be removed during audio processing
		'MIN_FREQUENCY': 20,
		# Frequencies above this value will be removed during audio processing
		'MAX_FREQUENCY': 10000,
		# Number of frequency bins to use when transforming audio to frequency domain
		'FFT_N_BINS': 40,
		#Length (ms) of the rolling audio window to be used. Will be adjusted to
		# improve fft performance.
		'FFT_WINDOW_LENGTH': 100
	}
}

# Return a dict of available visualizations
def visualizations(config):
	import numpy as np
	import dsp

	N_PIXELS = config['N_PIXELS']

	# A valid visualization is a function with the following signature:
	# Callable[[np.ndarray[float, FFT_N_BINS], np.ndarray[float, len(FFT_WINDOW)]]
	#  -> np.ndarray[int, 4, N_PIXELS]]

	def visualize_waveform(_, waveform, __):
		interpolated = dsp.interpolate(waveform, N_PIXELS)
		clipped = np.clip(interpolated - 0.5, 0, 1) * 50

		zeros = np.zeros(N_PIXELS);
		return np.array([clipped, clipped, clipped, zeros]);

	def visualize_spectrum(spectrum, _, __):
		interpolated = dsp.interpolate(spectrum, N_PIXELS)
		pixels = np.array([
			np.clip(1*np.log(interpolated*10), 0, 1),
			np.clip(0.3*np.log(interpolated*10), 0, 1),
			np.clip(0.3 * interpolated, 0, 1),
			np.tile(0, N_PIXELS),
		])
		return pixels * 255;

	smoothing = dsp.ExpFilter(np.tile(1e-1, N_PIXELS), alpha_decay=0.1, alpha_rise=0.7)
	def visualize_spectrum_smooth(spectrum, _, __):
		interpolated = dsp.interpolate(spectrum, N_PIXELS)
		interpolated = smoothing.update(interpolated)
		pixels = np.array([
			np.clip(1*np.log(interpolated*10), 0, 1),
			np.clip(0.3*np.log(interpolated*10), 0, 1),
			np.clip(0.3 * interpolated, 0, 1),
			np.tile(0, N_PIXELS),
		])
		return pixels * 255;

	def visualize_spectrum_2(y, _, __):
		interpolated = dsp.interpolate(y, N_PIXELS)
		log_part = np.log(interpolated*10)

		log_part /= 3
		log_part = 0.5 + np.clip(log_part, 0, 0.5)

		def color_from_value (x):
			return hsv2rgb(x, 1, x)

			colors = np.array([color_from_value(h) for h in log_part]).transpose()
			pixels = np.array([
				colors[0],
				colors[1],
				colors[2],
				np.clip(0.3 * interpolated, 0, 1),
			])
			return pixels * 255;

	indices = None
	fft_smoothing = dsp.ExpFilter(np.tile(1e-1, int(config['fft_samples_per_window'] / 2)), alpha_decay=0.1, alpha_rise=0.7)
	def folded_fourier(_, __, fft_data):
		nonlocal indices

		fft, freqs = fft_data
		output = np.zeros((4, N_PIXELS), dtype=np.float64)

		fft = np.log(fft / 100)

		if (indices == None):
			indices = list([0])
			f = 55
			done = False
			while not done:
				compatible = max([j for j, freq in enumerate(freqs) if freq < f])
				# print(f'{compatible} of {len(freqs)}')
				done = compatible == len(freqs) - 1
				f *= 2
				indices.append(compatible)

		# print(indices)

		for i in range(1, len(indices) - 1):
			prominence = abs(1/(0.49999 - (i / len(indices) - 1)))**2
			color = np.tile(hsv2rgb(i / len(indices), 1, prominence), (N_PIXELS, 1)).transpose()
			value = np.clip(dsp.interpolate(fft[indices[i-1]:indices[i]], N_PIXELS), 0, 1)
			fold = np.array([color[0] * value, color[1] * value, color[2] * value, np.zeros(N_PIXELS)])
			# print(f'shapes: {color.shape} and {value.shape} and {fold.shape}')
			# print(fold)
			output += fold

		return output * 255

	return {
		'spectrum': visualize_spectrum,
		'waveform': visualize_waveform,
		'spectrum2': visualize_spectrum_2,
		'smooth': visualize_spectrum_smooth,
		'folded': folded_fourier
	}
