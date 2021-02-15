from __future__ import print_function
from __future__ import division
import time
import numpy as np
import state

# Handler for audio samples. Will be updated in 'on_state_change'
process_sample = lambda samples: None

def on_state_change (config, visualization):
    # FFT window shape
    fft_window = np.hamming(config.fft_samples_per_window)

    # FFT binning (mel-bank-transformation)
    mel_trafo, (mel_x, _) = dsp.compute_melmat(
        num_mel_bands=config.FFT_N_BINS,
        freq_min=config.MIN_FREQUENCY,
        freq_max=config.MAX_FREQUENCY,
        num_fft_bands=int(config.fft_samples_per_window / 2),
        sample_rate=config.SAMPLE_RATE
    )

    #Smoothing on spectrum output
    mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.FFT_N_BINS), alpha_decay=0.8, alpha_rise=0.99)

    # Audio sample rolling window
    y_roll = np.random.rand(config.fft_samples_per_window) / 1e16

    def update(audio_samples):
        nonlocal y_roll
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
        mel = mel**2
        # mel = mel_smoothing.update(mel)
        led_output = visualization_effect(mel, y_data)

        t_vis = time.time() - start

        return ((y_data, mel, led_output), (t_roll, t_fft, t_vis))

    # Update sample handler
    process_sample = update

state.register_on_state_change_hander(on_state_change)
