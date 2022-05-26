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

def init(cfg, tick_callback):
    global app, view, input_curve, mel_curve, ready, config, fft_curves, fft_curve_pens, black_pen, data_lock
    config = cfg

    WAVES_LEN = config['WAVES_LEN']
    NUM_CURVES = config['NUM_CURVES']

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
        time.sleep(max(0, last_update - time.time()))

def update(output):
    global ready, input_curve, mel_curve, fft_curves, config, pa_idx, WAVES_LEN
    if (not ready or not config['USE_GUI']):
        return

    import pyqtgraph as pg

    data_waves, logger, _ = output

    with data_lock:
        for i, (x_data_wave, y_data_wave) in enumerate(data_waves):
            fft_curves[i].setData(y=y_data_wave, x=x_data_wave)

    logger('GUI_DATA')