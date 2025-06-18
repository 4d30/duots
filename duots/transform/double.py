#!/bin/env python

import itertools as its
import operator as op
import statistics as sts
import functools as fts
from scipy import signal
from scipy import fft


from transform import single as single


@fts.lru_cache(maxsize=128)
def passalong(signal_pair: tuple[tuple]) -> tuple[tuple]:
    signal_pair = map(single.passalong, signal_pair)
    signal_pair = tuple(signal_pair)
    return signal_pair


@fts.lru_cache(maxsize=128)
def dft(window_pairs: tuple[tuple]) -> tuple[tuple]:
    signal_pair = map(single.dft, window_pairs)
    signal_pair = tuple(signal_pair)
    return signal_pair


@fts.lru_cache(maxsize=128)
def autocorrelate(signal_pair: tuple[tuple]) -> tuple[tuple]:
    value = map(single.autocorrelate, signal_pair)
    value = tuple(value)
    return value


@fts.lru_cache(maxsize=128)
def zerosq(signal_pair: tuple[tuple]) -> tuple[tuple]:
    signal_pair = map(single.zerosq, signal_pair)
    signal_pair = tuple(signal_pair)
    return signal_pair


@fts.lru_cache(maxsize=128)
def findpeaks(signal_pair: tuple[tuple]) -> tuple[tuple]:
    signal_pair = map(single.findpeaks, signal_pair)
    signal_pair = tuple(signal_pair)
    return signal_pair


@fts.lru_cache(maxsize=128)
def cross_correlate(signal_pair: tuple[tuple]) -> tuple[tuple]:
    """ Cross-correlates two signals. The function first windows
    the signal, then calculates the cross correlation of each window,
    locates the max cross correlation in each window, and then selects
    the median value of the max cross correlations from the windows.

    Keyword arguments:
        signal_pair -- a tuple of two timeseries signals
    """
    sig_a = op.itemgetter(0)(signal_pair)
    sig_a = tuple(sig_a)
    sig_b = op.itemgetter(1)(signal_pair)
    sig_b = tuple(sig_b)
    corr = map(signal.correlate, sig_a, sig_b)
    corr = map(op.methodcaller('tolist'), corr)
    corr = map(tuple, corr)
    corr = tuple(corr)
    return corr
