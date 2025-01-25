from __future__ import print_function
from __future__ import division
import time
import numpy as np
import state
import dsp

# Handler for audio samples. Will be updated in 'on_state_change'
process_sample = lambda samples, latency, logger: None

def on_state_change (config, visualization):
    print('on_state_change')
    global process_sample
    # FFT window shape
    fft_window = np.blackman(config['fft_samples_per_window'])

    # FFT binning (mel-bank-transformation)
    mel_trafo, (mel_x, fft_x) = dsp.compute_melmat(
        num_mel_bands=config['FFT_N_BINS'],
        freq_min=config['MIN_FREQUENCY'],
        freq_max=config['MAX_FREQUENCY'],
        num_fft_bands=int(config['fft_samples_per_window'] // 2 + 1),
        sample_rate=config['SAMPLE_RATE']
    )

    #Smoothing on spectrum output
    mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config['FFT_N_BINS']), alpha_decay=0.8, alpha_rise=0.99)

    # Audio sample rolling window
    y_roll = np.random.rand(config['fft_samples_per_window']) / 1e16

    FFT_LEN = config['fft_samples_per_window'] // 2 + 1
    NUM_SPECIAL_CURVES = 2
    WAVES_LEN = config['WAVES_LEN']
    NUM_CURVES = config['NUM_CURVES']
    l_audio = config['fft_samples_per_window']
    pa_idx = 0

    len_wave = lambda i: WAVES_LEN - i * 200
    fft_filter = dsp.ExpFilter(np.tile(1e-1, FFT_LEN), alpha_decay=0.3, alpha_rise=0.3)
    window_inverse = 1/np.blackman(l_audio)
    curve_max_filter = [dsp.ExpFilter(1, alpha_decay=0.1, alpha_rise=0.8) for _ in range(NUM_CURVES - NUM_SPECIAL_CURVES)]


    def update(audio_samples, latency, logger):
        nonlocal y_roll
        logger('idle')

        # Normalize samples between 0 and 1
        y_update = audio_samples / 2.0**15
        # Construct a rolling window of audio samples
        l = len(y_update)
        # print((l, len(y_roll)))
        y_roll[:-l] = y_roll[l:]
        y_roll[-l:] = y_update
        y_data = y_roll.astype(np.float32)

        logger('roll')

        # Fourier transform and mel transformation
        N = len(y_data)
        # print(f'len(fft_): {len(fft_)}')
        fft = np.fft.rfft(y_data)
        fft_abs = np.abs(fft)
        fft_angle = np.angle(fft)
        mel = mel_trafo(fft_abs)
        # mel = mel**2

        logger('fft')

        # Visualize
        # mel = mel_smoothing.update(mel)
        led_output = visualization(mel, y_data, (fft_abs, fft_x))

        logger('vis')

        # Frequency weighing
        freqs = np.linspace(0, config['SAMPLE_RATE'] // 2, FFT_LEN)
        cutoff_freq_upper_hz = config['MAX_FREQUENCY']
        cutoff_freq_lower_hz = config['MIN_FREQUENCY']
        freq_to_samples = lambda f: int(config['SAMPLE_RATE'] // f)
        cutoff_idx_upper = np.argwhere(freqs > cutoff_freq_upper_hz)[0][0]
        cutoff_idx_lower = np.argwhere(freqs > cutoff_freq_lower_hz)[0][0]
        freq_weighing = np.concatenate((
            np.zeros(cutoff_idx_lower),
            np.linspace(1.5, 3.0, cutoff_idx_upper - cutoff_idx_lower),
            np.zeros(FFT_LEN - cutoff_idx_upper)))

        y = 5
        cutoff = [1, 14, 400]
        num_clip = [0, 3, 60]
        clip_sides = [num_clip[i] * freq_to_samples(freqs[cutoff[i]]) for i in range(len(cutoff))]
        scale = [1.5, 2, 3]
        NUM_ABOVE = 1
        NUM_BELOW = 1

        data_waves = [tuple()] * NUM_CURVES
        for i in range(NUM_CURVES - NUM_ABOVE - NUM_BELOW):

            fft = fft_abs.copy()
            fft[0] *= 0 if i < 1 else 0
            if (i > 0):
                a = cutoff[i]-1
                fft[1:cutoff[i]] = 0
            irfft = np.fft.irfft(fft)
            irfft = (irfft)[y:l_audio-y].clip(-1, 1) * scale[i] - 0.1
            l_irfft = len(irfft)
            n_clip = clip_sides[i]

            irfft = irfft[n_clip:l_irfft-n_clip]
            l_irfft -= 2 * n_clip
            audio_copy = np.zeros(l_irfft)
            audio_copy[:l_irfft // 2] = irfft[:l_irfft//2][::-1]
            audio_copy[l_irfft // 2:] = irfft[l_irfft//2:][::-1]

            y_data_wave = dsp.interpolate(audio_copy, len_wave(i+NUM_BELOW))
            y_data_wave_max = curve_max_filter[i].update(y_data_wave.max())
            y_data_wave /= max(0.02, min(3, y_data_wave_max + 0.5))
            y_data_wave = y_data_wave

            x_data_wave = np.arange((i + NUM_BELOW) * 100, WAVES_LEN - (i + NUM_BELOW) * 100)
            data_waves[i+NUM_BELOW] = (x_data_wave, y_data_wave)

        NM1 = NUM_CURVES - 1
        y_mel = dsp.interpolate(mel, WAVES_LEN - NM1 * 200) - 0.5
        x_mel = np.arange(NM1 * 100, WAVES_LEN - NM1 * 100)
        data_waves[NM1] = (x_mel, y_mel)

        NM2 = 0
        y_audio = dsp.interpolate(y_data, WAVES_LEN - NM2 * 200)
        x_audio = np.arange(NM2 * 100, WAVES_LEN - NM2 * 100)
        data_waves[NM2] = (x_audio, y_audio)

        logger('waves')

        return (data_waves, logger, led_output)

    # Update sample handler
    process_sample = update

state.register_on_state_change_hander(on_state_change)
