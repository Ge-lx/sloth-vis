import numpy as np
from utils import memoize

@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z

class ExpFilter:
    """Simple exponential smoothing filter"""
    def __init__(self, val=0.0, alpha_decay=0.5, alpha_rise=0.5):
        """Small rise / decay factors = more smoothing"""
        assert 0.0 < alpha_decay < 1.0, 'Invalid decay smoothing factor'
        assert 0.0 < alpha_rise < 1.0, 'Invalid rise smoothing factor'
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise
        self.value = val

    def update(self, value):
        if isinstance(self.value, (list, np.ndarray, tuple)):
            alpha = value - self.value
            alpha[alpha > 0.0] = self.alpha_rise
            alpha[alpha <= 0.0] = self.alpha_decay
        else:
            alpha = self.alpha_rise if value > self.value else self.alpha_decay
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value

scale = 200.0
hertz_to_mel = lambda freq: 2595.0 * np.log10(1 + (freq / scale))
mel_to_hertz = lambda mel: scale * (10**(mel / 2595.0)) - scale

def compute_melmat(num_mel_bands=12, freq_min=64, freq_max=8000, num_fft_bands=513, sample_rate=16000):
    mel_max = hertz_to_mel(freq_max)
    mel_min = hertz_to_mel(freq_min)
    delta_mel = np.abs(mel_max - mel_min) / (num_mel_bands + 1.0)
    frequencies_mel = mel_min + delta_mel * np.arange(0, num_mel_bands + 2)

    upper_edges_mel = frequencies_mel[2:]
    center_frequencies_mel = frequencies_mel[1:-1]

    center_frequencies_hz = mel_to_hertz(center_frequencies_mel)
    upper_edges_hz = mel_to_hertz(upper_edges_mel)

    freqs = np.linspace(0.0, sample_rate / 2.0, num_fft_bands)

    mean = lambda x: x.sum() if len(x) > 0 else 0
    ind_compat = lambda f_lim: [i for i, f in enumerate(freqs) if f < f_lim and f > freq_min]
    ind_low_lim = max([i for i, f in enumerate(freqs) if f <= freq_min])

    indices = [ind_compat(f) for f in upper_edges_hz]
    for i in range(len(indices)):
        comp = indices[i]
        if len(comp) == 0:
            if i == 0:
                indices[i] = ind_low_lim
            else:
                indices[i] = indices[i-1]
        else :
            indices[i] = max(comp)

    print(indices)
    freq_weight = lambda i: 1#(0.2 + (i+1)*4/num_mel_bands)

    def transform (fft_spectrum):
        bin_slice = lambda i: fft_spectrum[(indices[i-1] if i > 0 else ind_low_lim):indices[i]]

        i = 0
        copy = 0
        bin_means = np.zeros(num_mel_bands)
        while i < num_mel_bands:
            if indices[i] == (indices[i-1] if i > 0 else ind_low_lim):
                copy += 1
            elif copy:
                bin_means[i-copy:i+1] = mean(bin_slice(i)) * freq_weight(i) / copy
                copy = 0
            else:
                bin_means[i] = mean(bin_slice(i)) * freq_weight(i)
            i += 1

        [mean(bin_slice(i)) * freq_weight(i) for i in range(num_mel_bands)]
        return np.array(bin_means) / (num_fft_bands / 5)

    return transform, (center_frequencies_hz, freqs)