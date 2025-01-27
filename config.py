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

        # PulseAudio Configuration
		# ----------------------------------------------

		# Input and output have to use the same sample-rate and number of channels.
		# Any unavoidable differences should be dealt with using alsa's pcm_rate.
		# Hardware sample rate
		'SAMPLE_RATE': 44100,
		# Number of channels
		'CHANNELS': 2,

		# PulseAudio input mode. Choose from
		# ['default_sink', 'default_source', ' sink_by_name', 'source_by_name']
		'AUDIO_INPUT_MODE': 'default_source',

		# Full PulseAudio sink/source name.
		# Only used for ['sink_by_name', 'source_by_name']
		'AUDIO_INPUT_NAME': '',

		# LED Output
		# ----------------------------------
		# IP address(s) of the WLED ESP8266.
        'UDP_IP': ['192.168.0.147', '192.168.0.108'],
		# Port number used for socket communication between Python and ESP8266
		'UDP_PORT': 21324,
		# Number of pixels in the LED strip (should match WLED settigs)
        'N_PIXELS': [23, 39],
		# Target LED framerate. Will warn when this can't be met.
        'FPS_LED': 60,

		# FFT Settings 
		# ----------------------------------
		# Frequencies below this value will be removed during audio processing
		'MIN_FREQUENCY': 20,
		# Frequencies above this value will be removed during audio processing
        'MAX_FREQUENCY': 15000,
		# Number of frequency bins to use when transforming audio to frequency domain
		'FFT_N_BINS': 300,
		#Length (ms) of the rolling audio window to be used. Will be adjusted to
		# improve fft performance.
        'FFT_WINDOW_LENGTH': 40,

        'WAVES_LEN': 1800,
        'NUM_CURVES': 5

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


	blue_ish = np.array([0, 40, 255])
	purple_ish = np.array([120, 0, 230])
	orange_ish = np.array([255, 60, 40])
	white = np.array([255, 255, 255])
	black = np.array([0, 0, 0])
	state_adv_multi_strip = {
		'in_beat': False,
		'colors': [blue_ish, purple_ish, orange_ish],
		'color_idx_a': 0,
		'color_idx_b': 1,
		'hi_freq_region': [8e3, 12e3],
		'lo_freq_region': [0, 100],
		'color_brightness': dsp.ExpFilter(0, alpha_decay=0.1, alpha_rise=0.5),
		'lo_energy_mean': dsp.ExpFilter(0, alpha_decay=0.01, alpha_rise=0.01),
		'hi_energy_mean': dsp.ExpFilter(0, alpha_decay=0.001, alpha_rise=0.001),
		'segments': [5, 9], # 3 on small strip, 5 on large strip
		'mapping': {
			'lo': ([],[]),#([1], [2]),
			'hi': ([2], [1, 7]),
			'color_a': ([], [0, 2, 3, 4, 5, 6, 8]),
			'color_b': ([0, 1, 3, 4], [])
		}
	}

	def advanced_multi_strip (spectrum, waveform, fft_data):
		fft_vals, fft_freqs = fft_data
		lo_freq_reg = state_adv_multi_strip['lo_freq_region']
		hi_freq_reg = state_adv_multi_strip['hi_freq_region']
		colors = state_adv_multi_strip['colors']

		bins_lo = fft_vals[np.where(np.all([fft_freqs < lo_freq_reg[1], fft_freqs > lo_freq_reg[0]], axis=0))]
		bins_hi = fft_vals[np.where(np.all([fft_freqs < hi_freq_reg[1], fft_freqs > hi_freq_reg[0]], axis=0))]

		lo_mean = np.mean(bins_lo)
		hi_mean = np.mean(bins_hi)

		lo_filt = max(4, state_adv_multi_strip['lo_energy_mean'].update(lo_mean))
		hi_filt = max(4, state_adv_multi_strip['hi_energy_mean'].update(hi_mean))

		hi_bright = min(1, hi_mean / hi_filt) * 0.8
		in_beat = lo_mean >= lo_filt * 1.5

		new_beat = not state_adv_multi_strip['in_beat'] and in_beat
		if (not in_beat and state_adv_multi_strip['in_beat']):
			state_adv_multi_strip['in_beat'] = False
		if (new_beat):
			state_adv_multi_strip['in_beat'] = True
			state_adv_multi_strip['color_idx_a'] = state_adv_multi_strip['color_idx_b']
			state_adv_multi_strip['color_idx_b'] = (state_adv_multi_strip['color_idx_b'] + 1) % len(colors)

		color_a = colors[state_adv_multi_strip['color_idx_a']]
		color_b = colors[state_adv_multi_strip['color_idx_b']]
		color_brightness = state_adv_multi_strip['color_brightness'].update(0.6 if in_beat else 0.4)

		num_strips = len(state_adv_multi_strip['segments'])
		data_leds = []
		for i in range(num_strips):
			data_led = np.zeros((config['N_PIXELS'][i], 3), np.uint8)
			idx_range = np.arange(len(data_led))
			num_segs = state_adv_multi_strip['segments'][i]
			seg_length = len(data_led) / num_segs

			segment_idcs = [np.where(np.all([idx_range >= (i * seg_length), idx_range < ((i+1) * seg_length)], axis=0))[0] for i in range(num_segs)]

			lo_segs = state_adv_multi_strip['mapping']['lo'][i]
			hi_segs = state_adv_multi_strip['mapping']['hi'][i]
			color_a_segs = state_adv_multi_strip['mapping']['color_a'][i]
			color_b_segs = state_adv_multi_strip['mapping']['color_b'][i]

			for s in lo_segs:
				led_idcs = segment_idcs[s]
				val = white if in_beat else black
				data_led[led_idcs] = np.full((len(led_idcs), 3), val)

			for s in hi_segs:
				led_idcs = segment_idcs[s]
				val =  white * hi_bright
				data_led[led_idcs] = np.full((len(led_idcs), 3), val)

			for s in color_a_segs:
				led_idcs = segment_idcs[s]
				val =  color_a * color_brightness
				data_led[led_idcs] = np.full((len(led_idcs), 3), val)

			for s in color_b_segs:
				led_idcs = segment_idcs[s]
				val =  color_b * color_brightness
				data_led[led_idcs] = np.full((len(led_idcs), 3), val)

			data_leds.append(data_led)

		return data_leds

	return {
		'new': advanced_multi_strip,
		'smooth': visualize_spectrum_smooth,
		'waveform': visualize_waveform,
		'spectrum': visualize_spectrum,
		'spectrum2': visualize_spectrum_2,
		'folded': folded_fourier
	}
