from __future__ import print_function
from __future__ import division
import time
import numpy as np

ready = False
config = dict()

def init(cfg, tick_callback):
    global app, view, input_curve, mel_curve, ready, config
    config = cfg

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
    input_plot = layout.addPlot(title='Audio Input', colspan=3)
    input_plot.setRange(yRange=[-0.1, 1.2])
    input_plot.disableAutoRange(axis=pg.ViewBox.YAxis)

    x_data = np.array(range(1, config['fft_samples_per_window'] + 1))
    input_curve = pg.PlotCurveItem()
    input_curve.setData(x=x_data, y=x_data*0)
    input_plot.addItem(input_curve)

    # Mel filterbank plot
    fft_plot = layout.addPlot(title='Filterbank Output', colspan=3)
    fft_plot.setRange(yRange=[-0.1, 1.2])
    fft_plot.disableAutoRange(axis=pg.ViewBox.YAxis)

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
    global ready, input_curve, mel_curve, config
    if (not ready or not config['USE_GUI']):
        return

    audio, mel, _ = output
    input_curve.setData(y=audio)
    mel_curve.setData(y=mel)
