from __future__ import print_function
from __future__ import division
import time
import numpy as np
import dsp
import state
from scipy.signal import find_peaks
from threading import Lock


ready = False
data_lock = Lock()
config = state.default_config

WAVES_LEN = 1800
FFT_LEN = config['fft_samples_per_window'] // 2 + 1
NUM_CURVES = 5
NUM_SPECIAL_CURVES = 2
l_audio = config['fft_samples_per_window']
pa_idx = 0

len_wave = lambda i: WAVES_LEN - i * 200
fft_filter = dsp.ExpFilter(np.tile(1e-1, FFT_LEN), alpha_decay=0.3, alpha_rise=0.3)
window_inverse = 1/np.blackman(l_audio)
curves_smoothers = [{'a': 0, 'b': l_audio - 1, 's': dsp.ExpFilter(np.tile(0.5, l_audio), alpha_decay=0.2, alpha_rise=0.3)} for i in range(NUM_CURVES - NUM_SPECIAL_CURVES)]
curve_max_filter = [dsp.ExpFilter(1, alpha_decay=0.1, alpha_rise=0.8) for _ in range(NUM_CURVES - NUM_SPECIAL_CURVES)]

def init(cfg, tick_callback):
    global app, view, input_curve, mel_curve, ready, config, fft_curves, fft_curve_pens, black_pen, data_lock
    config = cfg
    # WAVES_LEN = 

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui, QtCore

    # Create GUI window
    app = QtGui.QApplication([])
    view = pg.GraphicsView()
    layout = pg.GraphicsLayout(border=(100,100,100))
    view.setCentralItem(layout)
    view.show()
    view.setWindowTitle('Visualization')
    view.resize(800,600)

    # FFT Plot
    plot_wave = layout.addPlot(title='Sloth 2', colspan=3)
    plot_wave.setRange(yRange=[1, 23], xRange=[0, WAVES_LEN])
    plot_wave.disableAutoRange(axis=pg.ViewBox.YAxis)
    plot_wave.disableAutoRange(axis=pg.ViewBox.XAxis)
    # plot_wave.setDownsampling(ds=True, auto=True, mode='subsample')
    # plot_wave.hideAxis('bottom')
    # plot_wave.hideAxis('left')

    black_pen = pg.mkPen((0, 0, 0, 1))
    fft_curve_pens = [pg.mkPen(color=pg.intColor(i), width=1) for i in range(NUM_CURVES)]
    fft_curves = [pg.PlotCurveItem(pen=pen, antialias=True, skipFiniteCheck=True) for pen in fft_curve_pens]
    for i, curve in enumerate(fft_curves):
        x_data_wave = np.arange(i * 100, WAVES_LEN - i * 100)
        curve.setData(x=x_data_wave, y=x_data_wave*0)
        plot_wave.addItem(curve)

    ready = True
    last_update = time.time()
    while ready:
        last_update = last_update + 1.0 / config['FPS_GUI']
        with data_lock:
            app.processEvents()
        tick_callback()
        # print(f'Sleeping {max(0, last_update - time.time()):3.3F}ms')
        time.sleep(max(0, last_update - time.time()))

def update(output):
    global ready, input_curve, mel_curve, fft_curves, config, pa_idx, WAVES_LEN
    if (not ready or not config['USE_GUI']):
        return

    import pyqtgraph as pg

    audio, mel, _, fft, logger, dt = output
    # l_audio = len(audio)
    pa_idx += dt
    fft = np.abs(fft)
    # fft_angle = np.angle(fft)
    # fft = fft_filter.update(fft)

    # Frequency weighing
    freqs = np.linspace(0, config['SAMPLE_RATE'] // 2, FFT_LEN)
    cutoff_freq_upper_hz = config['MAX_FREQUENCY']
    cutoff_freq_lower_hz = config['MIN_FREQUENCY']
    cutoff_idx_upper = np.argwhere(freqs > cutoff_freq_upper_hz)[0][0]
    cutoff_idx_lower = np.argwhere(freqs > cutoff_freq_lower_hz)[0][0]
    freq_weighing = np.concatenate((
        np.zeros(cutoff_idx_lower),
        np.linspace(1.5, 3.0, cutoff_idx_upper - cutoff_idx_lower),
        np.zeros(FFT_LEN - cutoff_idx_upper)))

    # Peak finding and sorting 
    # freq_peaks, _ = find_peaks(fft, height=5, prominence=0.3, distance=10, width=2)
    # prominence_sorted_indices = np.argsort(fft[freq_peaks] * freq_weighing[freq_peaks])
    # sorted_peaks = freqs[freq_peaks][prominence_sorted_indices][::-1][:NUM_CURVES - 2]
    # sorted_angles = fft_angle[freq_peaks][prominence_sorted_indices][::-1][:NUM_CURVES - 2]

    # freq_sorted_indices = np.argsort(sorted_peaks)
    # freq_sorted_peaks = sorted_peaks[freq_sorted_indices]
    # freq_sorted_angles = sorted_angles[freq_sorted_indices]

    # freq_idx_freq_sorted_indices = freq_peaks[prominence_sorted_indices][::-1][:NUM_CURVES - 2][freq_sorted_indices]
    # fft[freq_peaks] = 1000
    # fft[cutoff_idx_lower] = 1200
    # fft[cutoff_idx_upper] = 1200

    freq_to_samples = lambda f: int(config['SAMPLE_RATE'] // f)
    wave_offset = lambda i: np.log((3*i)**2 + 1) + 1.5**i + 3*i
    wave_scale = lambda i: ((i+2)**2) / (i+1) * 1.2
    cutoff = [1, 30, 300]
    num_clip = [0, 10, 60]
    clip_sides = [num_clip[i] * freq_to_samples(freqs[cutoff[i]]) for i in range(len(cutoff))]
    scale = [1.5, 2, 3]
    y = 5

    low_b = l_audio
    NUM_ABOVE = 1
    NUM_BELOW = 1
    for i in range(NUM_CURVES - NUM_ABOVE - NUM_BELOW):
        y_data_wave = np.full((len_wave(i+NUM_BELOW)), wave_offset(i+NUM_BELOW))

        fft = fft.copy()
        fft[0] /= scale[i]
        if (i > 0):
            a = cutoff[i]-1
            fft[1:cutoff[i]] = 0#*= np.hanning(a*2)[:a]
        irfft = np.fft.irfft(fft)
        irfft = (irfft)[y:l_audio-y].clip(-1, 1) * scale[i] - 0.1
        l_irfft = len(irfft)
        n_clip = clip_sides[i]

        irfft = irfft[n_clip:l_irfft-n_clip]
        l_irfft -= 2 * n_clip
        audio_copy = np.zeros(l_irfft)
        audio_copy[:l_irfft // 2] = irfft[:l_irfft//2][::-1]
        audio_copy[l_irfft // 2:] = irfft[l_irfft//2:][::-1]

        y_data_wave = dsp.interpolate(audio_copy, len_wave(i+NUM_BELOW)) # audio_tirggered[a-b:]
        # y_data_wave = np.sin(np.linspace(s / samp * 2 *np.pi, 2 * n * np.pi, len_wave(i)))
        y_data_wave_max = curve_max_filter[i].update(y_data_wave.max())
        y_data_wave /= max(0.02, min(3, y_data_wave_max + 0.5))
        y_data_wave = y_data_wave * wave_scale(i+NUM_BELOW) + wave_offset(i+NUM_BELOW)


        x_data_wave = np.arange((i + NUM_BELOW) * 100, WAVES_LEN - (i + NUM_BELOW) * 100)
        with data_lock:
            fft_curves[i+NUM_BELOW].setData(y=y_data_wave, x=x_data_wave)

    NM1 = NUM_CURVES - 1
    y_mel = dsp.interpolate(mel, WAVES_LEN - NM1 * 200) * 5 + wave_offset(NM1) - 3
    x_mel = np.arange(NM1 * 100, WAVES_LEN - NM1 * 100)
    with data_lock:
        fft_curves[NM1].setData(y=y_mel, x=x_mel)

    NM2 = 0
    y_audio = dsp.interpolate(audio, WAVES_LEN - NM2 * 200) * 2 + wave_offset(NM2) + 0.8
    x_audio = np.arange(NM2 * 100, WAVES_LEN - NM2 * 100)
    with data_lock:
        fft_curves[NM2].setData(y=y_audio, x=x_audio)

    logger('trigger')

        # if (True or i < num_peaks):
        #     # f_idx = freq_idx_freq_sorted_indices[i]

        #     # f = freq_sorted_peaks[i]
        #     # phi = freq_sorted_angles[i]
        #     samp = freq_to_samples(f)
        #     y = 5#samp // 2
        #     # o = phi / (2*np.pi) + 1
        #     # s = int(o * samp) # pa_idx % samp if i > 2 else 
        #     m = (l_audio - 2*y) / samp - 1.5
        #     n = min(int(m), 5 + 5 * i)
        #     # print((pa_idx % samp) / samp) - 0.5

        #     c = l_audio // 2 - y
        #     b = c + int(n/2) * samp
        #     a = c - int(n/2) * samp

        #     # b = int(l_audio - s)
        #     # offset = 0
        #     # if (b < low_b):
        #     #     low_b = b
        #     # else:
        #     #     offset = int((l_audio - low_b) / samp + 1)
        #     #     m -= offset
        #     #     b -= offset * samp
        #     #     if (b < low_b):
        #     #         low_b = b

        #     # a = int(l_audio - (n + offset) * samp  - s)
        #     # a = int((2*n+1) * samp + s)
        #     # b = s

        #     print(a, b, l_audio, m)
        #     if (True or b - a) > 0:
        #         fft = fft.copy()
        #         if (i > 0):
        #             fft[1:cutoff[i]] = 0
        #         # fft[-1000:] = 0
        #         irfft = np.fft.irfft(fft)[y:l_audio-y]
        #         l_irfft = len(irfft)
        #         audio_copy = np.zeros(l_irfft)
        #         audio_copy[:l_irfft // 2] = irfft[:l_irfft//2][::-1]
        #         audio_copy[l_irfft // 2:] = irfft[l_irfft//2:][::-1]
        #         # else:
        #         #     audio_copy = audio.copy()
        #         # for o in range(i):
        #         #         s_a = curves_smoothers[o]['a']
        #         #         s_b = curves_smoothers[o]['b']
        #         #         s_f = curves_smoothers[o]['s']
        #         #         val = s_f.value[s_a-s_b:] - 0.5
        #                 # audio_copy[:s_a] = np.tile(0.5, s_a)
        #                 # audio_copy[s_a:s_b] -= val

        #         # audio_tirggered = np.zeros_like(audio)
        #         # audio_tirggered[l_audio - b:] = audio_copy[:b]
        #         # curves_smoothers[i]['s'].update(audio_tirggered)
        #         # curves_smoothers[i]['a'] = a
        #         # curves_smoothers[i]['b'] = b

        #         y_data_wave = dsp.interpolate(audio_copy, len_wave(i)) # audio_tirggered[a-b:]
        #         # y_data_wave = np.sin(np.linspace(s / samp * 2 *np.pi, 2 * n * np.pi, len_wave(i)))
        #         y_data_wave_max = curve_max_filter[i].update(y_data_wave.max())
        #         y_data_wave /= max(0.02, min(1, y_data_wave_max + 0.5))
        #         y_data_wave = y_data_wave * wave_scale(i) + wave_offset(i)

            # if (i == 0):
            #     phi_str = f'{phi:3.2F}'
            #     print(f'Locked to f = {f:4.1F}Hz with phi = {phi_str:5} | s = {s:4.0F} | m = {m:1.2F} | a = {a:1.0F} | b = {b:1.0F}\n\n')
