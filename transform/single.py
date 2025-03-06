#!/bin/env python

import itertools as its
import operator as op
import statistics as sts

from scipy import fft
from scipy import signal as sig

from array import array


def passalong(signal: tuple[array]) -> tuple[array]:
    return signal

def dft(windows: tuple[array]) -> tuple[array]:

    # Subtract the median value from each window
    median = its.chain.from_iterable(windows)
    median = sts.median(median)
    medians = its.repeat(median, len(windows[0]))
    medians = map(its.repeat, medians, its.repeat(len(windows)))
    windows = map(lambda w, m: map(op.sub, w, m), windows, medians)
    windows = map(array, its.repeat('d'), windows)
    windows = tuple(windows)

    # Apply the Hann window
    hann_win = sig.windows.hann(len(windows[0]))
    hann_win_a = map(map,
                     its.repeat(op.mul),
                     windows,
                     its.repeat(hann_win))
    hann_win_a = map(array, its.repeat('d'), hann_win_a)
    hann_win_a = tuple(hann_win_a)

    # Do the thing
    dfts = map(fft.rfft, hann_win_a)
    dfts = map(map, its.repeat(abs), dfts)
    dfts = map(array, its.repeat('d'), dfts)
    dfts = tuple(dfts)

    freqs = fft.rfftfreq(len(windows[0]), d=1.0/100.0)
    freqs = array('d', freqs)

    data = map(zip, its.repeat(freqs), dfts)
    data = map(dict, data)
    data = tuple(data)

    return dfts


def autocorrelate(signal: tuple[array]) -> tuple[array]:
    corr = map(sig.correlate, signal, signal)
    corr = map(op.methodcaller('tolist'), corr)
    corr = map(array, its.repeat('d'), corr)
    corr = tuple(corr)
    return corr


def zerosq(signal: array) -> array:
    def zq(signal):
        signal = map(op.sub, signal, its.repeat(sts.median(signal)))
        signal = map(pow, signal, its.repeat(2))
        signal = array('d', signal)
        return signal
    signal = map(zq, signal)
    signal = tuple(signal)
    return signal


def findpeaks(signal: tuple[array]) -> tuple[array]:
    def fp(signal):
        pk_ht = sts.median(signal)
        peaks, _ = sig.find_peaks(signal,
                                  height=pk_ht,
                                  prominence=1)
        if not peaks.any():
            return array('d', (float('nan'),
                               float('nan'),))
        peaks = op.itemgetter(*peaks)(signal)
        if not hasattr(peaks, '__len__'):
            peaks = (peaks,)
        peaks = array('d', peaks)
        return peaks

    peaks = map(fp, signal)
    peaks = tuple(peaks)
    return peaks
