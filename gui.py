from __future__ import print_function
from __future__ import division
import time
import numpy as np
import dsp
import state
from scipy.signal import find_peaks

ready = False
config = state.default_config

FFT_LEN = config['fft_samples_per_window'] // 2 + 1
counter = 0

fft_filter = dsp.ExpFilter(np.tile(1e-1, int(config['fft_samples_per_window'] / 2)), alpha_decay=0.8, alpha_rise=0.99)

def init(cfg, tick_callback):
    global app, view, input_curve, mel_curve, ready, config, fft_curve
    config = cfg
    FFT_LEN = config['fft_samples_per_window'] // 2 + 1

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
    input_plot = layout.addPlot(title='Audio Input', colspan=1)
    input_plot.setRange(yRange=[-0.1, 240], xRange=[0, 1000])
    input_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
    input_plot.disableAutoRange(axis=pg.ViewBox.XAxis)

    x_data = np.array(range(1, 2 * (config['fft_samples_per_window'] + 1)))
    input_curve = pg.PlotCurveItem()
    input_curve.setData(x=x_data, y=x_data*0)
    input_plot.addItem(input_curve)

    # FFT Plot
    fft_plot_2 = layout.addPlot(title='Audio Input', colspan=2)
    fft_plot_2.setRange(yRange=[-1, 2.6], xRange=[0, 4000])
    fft_plot_2.disableAutoRange(axis=pg.ViewBox.YAxis)
    fft_plot_2.disableAutoRange(axis=pg.ViewBox.XAxis)

    x_data_fft = np.array(range(FFT_LEN))
    fft_curve = pg.PlotCurveItem()
    fft_curve.setData(x=x_data_fft, y=x_data_fft*0)
    fft_plot_2.addItem(fft_curve)

    # Mel filterbank plot
    fft_plot = layout.addPlot(title='Filterbank Output', colspan=1)
    fft_plot.setRange(yRange=[-0.1, 1.2], xRange=[0, 100])
    fft_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
    fft_plot.disableAutoRange(axis=pg.ViewBox.XAxis)

    x_data = np.array(range(1, config['FFT_N_BINS'] + 1))
    mel_curve = pg.PlotCurveItem()
    mel_curve.setData(x=x_data, y=x_data*0)
    fft_plot.addItem(mel_curve)

    ready = True
    last_update = time.time()
    while ready:
        last_update = last_update + 1 / config['FPS_GUI']
        app.processEvents()
        tick_callback()
        time.sleep(max(0, last_update - time.time()))

def update(output):
    global ready, input_curve, mel_curve, fft_curve, config, counter, FFT_LEN
    if (not ready or not config['USE_GUI']):
        return

    audio, mel, _, fft_, idx = output
    l_audio = len(audio)
    # counter += dt
    # counter %= l_audio
    # fft_ = fft_filter.update(fft_)

    print(counter)

    FFT_LEN_ = FFT_LEN * 4
    fft_ = dsp.interpolate(fft_, FFT_LEN_)

    debug = np.full((FFT_LEN), 0.5)
    o = 0
    def display (x):
        nonlocal o
        l = len(x)
        if (len(x) > len(debug) - o):
            return
        debug[o:o+l] = x
        o += l


    freqs = np.linspace(0, config['SAMPLE_RATE'] // 2, FFT_LEN_)
    # fft_lower_cutoff = np.argwhere(freqs > config['MIN_FREQUENCY'])[0,0]
    # print(f'fft_lower_cutoff: {fft_lower_cutoff}')
    # fft = fft_[fft_lower_cutoff:]
    fft = fft_
    freq_peaks, _ = find_peaks(fft, height=10, prominence=0.3, distance=5)
    # freq_peaks += fft_lower_cutoff
    # print(freq_peaks + fft_lower_cutoff)

    # fft_[freq_peaks + fft_lower_cutoff] = 10000
    # display(fft)


    # freq_peaks += fft_lower_cutoff
    cutoff_freq_hz = 1000
    cutoff_idx = np.argwhere(freqs > cutoff_freq_hz)[0][0]

    freq_weighing = np.concatenate((np.linspace(2, 0.5, cutoff_idx), np.zeros(FFT_LEN_ - cutoff_idx)))

    # freqs = 

    sorted_peaks = freqs[freq_peaks][np.argsort(fft[freq_peaks] * freq_weighing[freq_peaks])][::-1]
    fft[freq_peaks] = 10000
    print(sorted_peaks[:5])
    freq_to_samples = lambda f: int(config['SAMPLE_RATE'] // f)
    if len(sorted_peaks > 0):
        f = sorted_peaks[0]
        # for i, f in enumerate(freq_peaks[1:2]):
            # if f_factor < 1:
            #     audio_slice = audio[-(l_audio//2):]
            # else:
            #     a = l_audio // f_factor
            # f = 100
        samp = freq_to_samples(f)
        s = idx % samp
        x = int(1 * samp + s)
        print(f, s, x)
        audio_slice = audio[-x:-s]
        if (len(audio_slice) > 0):
            audio_slice = dsp.interpolate(audio_slice, FFT_LEN)
            display(audio_slice)
    # print(' ')

    # fft[idxs[-1:]] = 5000
    # display(audio)

    input_curve.setData(y=fft_, x=freqs)
    fft_curve.setData(y=audio)
    mel_curve.setData(y=mel)

