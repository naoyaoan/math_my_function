# -*- coding: utf-8 -*-
import cis
import numpy as np

def add_vibrato(file, fv, p=0.1):
    """
    与えられた音源にビブラートをかける関数

    file(wav): ビブラートをかけるwavfileを指定
    fv(int): 周波数がfvの正弦波の形で変動する
    p(int): 元の音源の周波数をa,上下させる周波数をbとした時,p=b/aとしたもの
            元の音源の周波数の何倍変動させるか指定する 
    """
    y,fs = cis.wavread(file)
    t = np.arange(0, y.shape[0]/fs, 1/fs)
    g = t - p/(2*np.pi*fv)*np.cos(2*np.pi*fv*t)
    passing = np.argwhere(g > t[-1])
    if passing.size != 0:
        g = g[:passing[0, 0]]
    mody = np.interp(g, t, y)
    cis.audioplay(mody, fs)
