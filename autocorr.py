# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as mlb
import numpy.linalg as nla
import scipy.signal as ss
import scipy.fftpack as sfft

# 自己相関係数を用いて基本周波数を返す
def autocorr(wave, fs, order=10, output='hertz'):
    wavec = np.correlate(wave, wave, 'fill')
    wave1c = wavec/np.max(wavec)
    median = round(len(wave1c)/2) - 1
    wave_ev = ss.argrelmax(wave1c, order=order) # 極値を求める
    while len(wave_ev[0]) == 1:
        if order == 0:
            break
        wave_ev = ss.argrelmax(wave1c, order=order)
        order = order - 1

    if len(wave_ev[0]) == 1:
        return None
    else:
        ev_loc = wave_ev[0][(wave_ev[0]>median).nonzero()]
        period = ev_loc[np.argmax(wave1c[ev_loc])]-median
        f0 = period*1/fs # 基本周波数[s]-> 単位が時間(一周期にf0秒)なのでHzに変換する
        if output == 'hertz':
            f0=1/f0 # 逆数にすることでHzに変換される(1秒間に何周するか)
            return f0
        elif output == 'time':
            return f0


def origin_dft(data):
    length = len(data)
    frequency_data_real, frequency_data_imag = [], []
    for k in range(length):
        b = 0j
        for n in range(length):
            b += data[n] * np.exp(-2j*np.pi*n*k/length)
        frequency_data_real.append(b.real)
        frequency_data_imag.append(b.imag)
    return frequency_data_real, frequency_data_imag
