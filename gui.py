from __future__ import print_function
from __future__ import division
import time
import numpy as np
import dsp
import state
from scipy.signal import find_peaks

ready = False
config = state.default_config

WAVES_LEN = 1920
NUM_CURVES = 5

fft_filter = dsp.ExpFilter(np.tile(1e-1, WAVES_LEN), alpha_decay=0.9, alpha_rise=0.5)

def init(cfg, tick_callback):
    global app, view, input_curve, mel_curve, ready, config, fft_curves, fft_curve_pens, black_pen
    config = cfg
    # WAVES_LEN = config['fft_samples_per_window'] // 2 + 1

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
    plot_wave = layout.addPlot(title='Audio Input', colspan=3)
    plot_wave.setRange(yRange=[-1, 11.8], xRange=[0, WAVES_LEN])
    plot_wave.disableAutoRange(axis=pg.ViewBox.YAxis)
    plot_wave.disableAutoRange(axis=pg.ViewBox.XAxis)

    black_pen = pg.mkPen((0, 0, 0, 1))
    fft_curve_pens = [pg.mkPen(color=pg.intColor(i), width=1) for i in range(NUM_CURVES)]
    fft_curves = [pg.PlotCurveItem(pen=pen, antialias=True, skipFiniteCheck=True) for pen in fft_curve_pens]
    for i, curve in enumerate(fft_curves):
        x_data_wave = np.arange(i * 100, WAVES_LEN - i * 100)
        curve.setData(x=x_data_wave, y=x_data_wave*0)
        plot_wave.addItem(curve)


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
    global ready, input_curve, mel_curve, fft_curves, config, pa_idx, WAVES_LEN
    if (not ready or not config['USE_GUI']):
        return

    import pyqtgraph as pg

    audio, mel, _, fft_, fft_angle, dt = output
    l_audio = len(audio)
    # pa_idx += dt
    fft_ = dsp.interpolate(fft_, WAVES_LEN)
    # fft_ = fft_filter.update(fft_)
    fft = fft_.copy()

    # Frequency weighing
    freqs = np.linspace(0, config['SAMPLE_RATE'] // 2, WAVES_LEN)
    cutoff_freq_upper_hz = 5000
    cutoff_freq_lower_hz = 20 # config['MIN_FREQUENCY'];
    cutoff_idx_upper = np.argwhere(freqs > cutoff_freq_upper_hz)[0][0]
    cutoff_idx_lower = np.argwhere(freqs > cutoff_freq_lower_hz)[0][0]
    freq_weighing = np.concatenate((
        np.zeros(cutoff_idx_lower),
        np.linspace(1.5, 0.8, cutoff_idx_upper - cutoff_idx_lower),
        np.zeros(WAVES_LEN - cutoff_idx_upper)))

    # Peak finding and sorting 
    freq_peaks, _ = find_peaks(fft, height=10, prominence=0.3, distance=5)
    prominence_sorted_indices = np.argsort(fft[freq_peaks] * freq_weighing[freq_peaks])
    sorted_peaks = freqs[freq_peaks][prominence_sorted_indices][::-1][:NUM_CURVES]
    sorted_angles = fft_angle[freq_peaks][prominence_sorted_indices][::-1][:NUM_CURVES]

    freq_sorted_indices = np.argsort(sorted_peaks)
    freq_sorted_peaks = sorted_peaks[freq_sorted_indices]
    freq_sorted_angles = sorted_angles[freq_sorted_indices]
    # fft[freq_peaks] = 1000
    # fft[cutoff_idx_lower] = 1200
    # fft[cutoff_idx_upper] = 1200


    if (len(freq_sorted_peaks) > 0):
        fft_peaks_str = ', '.join([f'{f:4.1F}' for f in freq_sorted_peaks[:5]])
        print(f'Weighted FFT peaks: [{fft_peaks_str}]')

    freq_to_samples = lambda f: int(config['SAMPLE_RATE'] // f)
    num_peaks = len(freq_sorted_peaks[:NUM_CURVES])
    for i in range(NUM_CURVES):
        waves = np.array([])
        len_wave = WAVES_LEN - i * 200
        scale_wave = 4 / ((i+0.5) * 2)
        offset_wave = 0.5 + np.log((5*i)**2 + 0.5) + 1 * i
        if (i >= num_peaks):
            fft_curves[i].setData(
                y=np.full((len_wave), 0.5 * scale_wave + offset_wave),
                pen=black_pen)
            continue

        f = freq_sorted_peaks[i]
        phi = freq_sorted_angles[i]
        samp = freq_to_samples(f)
        m = float(l_audio) / (samp) - 2
        n = min(m, 5)
        o = phi / (2*np.pi)
        s = int(o * samp)
        a = -int(n * samp + s)
        b = -(2*samp + s)
        audio_slice = audio[a:b]

        waves = dsp.interpolate(audio_slice, len_wave) * scale_wave + offset_wave
        fft_curves[i].setData(y=waves, pen=fft_curve_pens[i])

        if (i == 0):
            phi_str = f'{phi:3.2F}'
            print(f'Locked to f = {f:4.1F}Hz with phi = {phi_str:5} | s = {s:4.0F} | m = {m:1.2F} | a = {a:1.0F} | b = {b:1.0F}\n\n')
            # print(pa_idx)

    # input_curve.setData(y=fft, x=freqs)
    # mel_curve.setData(y=mel)

