from __future__ import print_function
from __future__ import division
import time
import numpy as np
import dsp
import state
from scipy.signal import find_peaks

ready = False
config = state.default_config

FFT_LEN_INTERPOLATION = 1 * (config['fft_samples_per_window'] // 2 + 1)
pa_idx = 0

fft_filter = dsp.ExpFilter(np.tile(1e-1, FFT_LEN_INTERPOLATION), alpha_decay=0.3, alpha_rise=0.55)

def init(cfg, tick_callback):
    global app, view, input_curve, mel_curve, ready, config, fft_curve
    config = cfg
    # FFT_LEN_INTERPOLATION = config['fft_samples_per_window'] // 2 + 1

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


    # Audio input plot
    # input_plot = layout.addPlot(title='Audio Input', colspan=1)
    # input_plot.setRange(yRange=[-0.1, 240], xRange=[0, 1000])
    # input_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
    # input_plot.disableAutoRange(axis=pg.ViewBox.XAxis)

    # x_data = np.array(range(1, 2 * (config['fft_samples_per_window'] + 1)))
    # input_curve = pg.PlotCurveItem()
    # input_curve.setData(x=x_data, y=x_data*0)
    # input_plot.addItem(input_curve)

    # FFT Plot
    fft_plot_2 = layout.addPlot(title='Audio Input', colspan=3)
    fft_plot_2.setRange(yRange=[-0.2, 1.2], xRange=[-200, FFT_LEN_INTERPOLATION + 200])
    fft_plot_2.disableAutoRange(axis=pg.ViewBox.YAxis)
    fft_plot_2.disableAutoRange(axis=pg.ViewBox.XAxis)

    x_data_fft = np.array(range(FFT_LEN_INTERPOLATION))
    fft_curve = pg.PlotCurveItem()
    fft_curve.setData(x=x_data_fft, y=x_data_fft*0)
    fft_plot_2.addItem(fft_curve)

    # Mel filterbank plot
    # fft_plot = layout.addPlot(title='Filterbank Output', colspan=1)
    # fft_plot.setRange(yRange=[-0.1, 1.2], xRange=[0, 100])
    # fft_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
    # fft_plot.disableAutoRange(axis=pg.ViewBox.XAxis)

    # x_data = np.array(range(1, config['FFT_N_BINS'] + 1))
    # mel_curve = pg.PlotCurveItem()
    # mel_curve.setData(x=x_data, y=x_data*0)
    # fft_plot.addItem(mel_curve)

    ready = True
    last_update = time.time()
    while ready:
        last_update = last_update + 1 / config['FPS_GUI']
        app.processEvents()
        tick_callback()
        time.sleep(max(0, last_update - time.time()))

def update(output):
    global ready, input_curve, mel_curve, fft_curve, config, pa_idx, FFT_LEN_INTERPOLATION
    if (not ready or not config['USE_GUI']):
        return

    audio, mel, _, fft_, fft_angle, dt = output
    l_audio = len(audio)
    # pa_idx += dt
    debug = np.full((FFT_LEN_INTERPOLATION), 0.5)
    # fft_ = dsp.interpolate(fft_, FFT_LEN_INTERPOLATION)
    # fft_ = fft_filter.update(fft_)
    fft = fft_.copy()

    # Frequency weighing
    freqs = np.linspace(0, config['SAMPLE_RATE'] // 2, FFT_LEN_INTERPOLATION)
    cutoff_freq_upper_hz = 5000
    cutoff_freq_lower_hz = 20 # config['MIN_FREQUENCY'];
    cutoff_idx_upper = np.argwhere(freqs > cutoff_freq_upper_hz)[0][0]
    cutoff_idx_lower = np.argwhere(freqs > cutoff_freq_lower_hz)[0][0]
    freq_weighing = np.concatenate((
        np.zeros(cutoff_idx_lower),
        np.linspace(1, 0.5, cutoff_idx_upper - cutoff_idx_lower),
        np.zeros(FFT_LEN_INTERPOLATION - cutoff_idx_upper)))

    # Peak finding and sorting 
    freq_peaks, _ = find_peaks(fft, height=10, prominence=0.3, distance=5)
    indices = np.argsort(fft[freq_peaks] * freq_weighing[freq_peaks])
    sorted_peaks = freqs[freq_peaks][indices][::-1]
    sorted_angles = fft_angle[freq_peaks][indices][::-1]
    # fft[freq_peaks] = 1000
    # fft[cutoff_idx_lower] = 1200
    # fft[cutoff_idx_upper] = 1200


    if (len(sorted_peaks) > 0):
        fft_peaks_str = ', '.join([f'{f:4.1F}' for f in sorted_peaks[:5]])
        print(f'Weighted FFT peaks: [{fft_peaks_str}]')

    freq_to_samples = lambda f: int(config['SAMPLE_RATE'] // f)
    if len(sorted_peaks > 0):
        idx = 0
        f = sorted_peaks[idx]
        phi = sorted_angles[idx]

        samp = freq_to_samples(f)
        m = float(l_audio) / (samp) - 2
        n = min(m, 6)

        o = phi / (2*np.pi)
        s = int(o * samp)

        a = -int(n * samp + s)
        b = -(2*samp + s)

        audio_slice = audio[a:b]

        phi_str = f'{phi:3.2F}'
        print(f'Locked to f = {f:4.1F}Hz with phi = {phi_str:5} | s = {s:4.0F} | m = {m:1.2F} | a = {a:1.0F} | b = {b:1.0F}\n\n')
        # print(pa_idx)

        if (len(audio_slice) > 0):
            debug = dsp.interpolate(audio_slice, FFT_LEN_INTERPOLATION)

    # input_curve.setData(y=fft, x=freqs)
    fft_curve.setData(y=debug)
    # mel_curve.setData(y=mel)

