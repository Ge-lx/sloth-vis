from __future__ import print_function
from __future__ import division
import time
import numpy as np

import config
import dsp
from utils import hsv2rgb

mel_trafo, (mel_x, _) = dsp.compute_melmat(
    num_mel_bands=config.FFT_N_BINS,
    freq_min=config.MIN_FREQUENCY,
    freq_max=config.MAX_FREQUENCY,
    num_fft_bands=int(config.fft_samples_per_window / 2),
    sample_rate=config.SAMPLE_RATE
)

# The fft window shape
fft_window = np.hamming(config.fft_samples_per_window)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.FFT_N_BINS), alpha_decay=0.8, alpha_rise=0.79)
maximum_filter = dsp.ExpFilter(np.tile(0, 1), alpha_decay=0.001, alpha_rise=0.99)

def visualize_waveform(_, waveform):
    interpolated = dsp.interpolate(waveform, config.N_PIXELS)
    clipped = np.clip(interpolated - 0.5, 0, 1) * 50

    zeros = np.zeros(config.N_PIXELS);
    return np.array([zeros, zeros, zeros, clipped]);

def visualize_spectrum(y, _):
    interpolated = dsp.interpolate(y, config.N_PIXELS)
    log_part = np.log(interpolated*10)

    # print(f'max: {} , min: {np.min(log_part)}')

    max_filt = 3 # maximum_filter.update([np.max(log_part)])[0]
    log_part /= 3
    log_part = 0.5 + np.clip(log_part, 0, 0.5)
    def color_from_value (x):
        return hsv2rgb(x, 1, x)

    colors = np.array([color_from_value(h) for h in log_part]).transpose()

    # print(log_part[123])

    pixels = np.array([
        # np.clip(1*log_part, 0, 1),
        # np.clip(0.3*log_part, 0, 1),
        # np.tile(0, config.N_PIXELS),
        colors[0],
        colors[1],
        colors[2],
        np.clip(0.3 * interpolated, 0, 1),
    ])
    return pixels * 255;

# Visualization effect to display on the LED strip
visualization_effect = visualize_spectrum


# Array containing the rolling audio sample window
y_roll = np.random.rand(config.fft_samples_per_window) / 1e16

def update(audio_samples):
    global y_roll
    start = time.time()

    # Normalize samples between 0 and 1
    y_update = audio_samples / 2.0**15 + 0.5
    # Construct a rolling window of audio samples
    l = len(y_update)
    y_roll[:-l] = y_roll[l:]
    y_roll[-l:] = y_update
    y_data = y_roll.astype(np.float32)

    t_roll = time.time() - start

    # Fourier transform and mel transformation
    N = len(y_data)
    YS = np.abs(np.fft.rfft(y_data * fft_window)[:N // 2])
    mel =  mel_trafo(YS)

    t_fft = time.time() - start

    # Visualize    
    mel = mel / 550
    mel = mel_smoothing.update(mel)
    led_output = visualization_effect(mel, y_data)

    t_vis = time.time() - start

    return ((y_data, mel, led_output), (t_roll, t_fft, t_vis))
