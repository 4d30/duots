#!/bin/env python

import itertools as its
import operator as op
import statistics as sts

from scipy import fft
from scipy import signal as sig


def passalong(signal: tuple[tuple]) -> tuple[tuple]:
    return signal


def dft(windows_in: tuple[tuple]) -> tuple[tuple]:

    # Subtract the median value from each window
    median = its.chain.from_iterable(windows_in)
    median = sts.median(median)
    medians = its.repeat(median, len(windows_in[0]))
    medians = map(its.repeat, medians, its.repeat(len(windows_in)))
    medians = map(tuple, medians)
    medians = zip(*medians)
    windows = map(lambda w, m: map(op.sub, w, m), windows_in, medians)
    windows = map(tuple, windows)
    windows = tuple(windows)

    # Apply the Hann window
    hann_win = sig.windows.hann(len(windows[0]))
    hann_win_a = map(map,
                     its.repeat(op.mul),
                     windows,
                     its.repeat(hann_win))
    hann_win_a = map(tuple, hann_win_a)
    hann_win_a = tuple(hann_win_a)

    # Do the thing
    dfts = map(fft.rfft, hann_win_a)
    dfts = map(map, its.repeat(abs), dfts)
    dfts = map(tuple, dfts)
    dfts = tuple(dfts)

    # freqs = fft.rfftfreq(len(windows[0]), d=1.0/100.0)
    # freqs = tuple('d', freqs)

    # data = map(zip, its.repeat(freqs), dfts)
    # data = map(dict, data)
    # data = tuple(data)
    return dfts


def autocorrelate(signal: tuple[tuple]) -> tuple[tuple]:
    corr = map(sig.correlate, signal, signal)
    corr = map(op.methodcaller('tolist'), corr)
    corr = map(tuple, corr)
    corr = tuple(corr)
    return corr


def zerosq(signal: tuple) -> tuple:
    def zq(signal):
        signal = map(op.sub, signal, its.repeat(sts.median(signal)))
        signal = map(pow, signal, its.repeat(2))
        signal = tuple(signal)
        return signal
    signal = map(zq, signal)
    signal = tuple(signal)
    return signal


def findpeaks(signal: tuple[tuple]) -> tuple[tuple]:
    def fp(signal):
        pk_ht = sts.median(signal)
        peaks, _ = sig.find_peaks(signal,
                                  height=pk_ht,
                                  prominence=1)
        if not peaks.any():
            return tuple((float('nan'), float('nan'),))
        peaks = op.itemgetter(*peaks)(signal)
        if not hasattr(peaks, '__len__'):
            peaks = (peaks,)
        peaks = tuple(peaks)
        return peaks

    peaks = map(fp, signal)
    peaks = tuple(peaks)
    return peaks
