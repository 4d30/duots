#!/bin/env python

import operator as op
import itertools as its
import statistics as sts
from array import array
import functools as fts

from calculators import single as single


def __transpose(signal_pair):
    # aa = op.itemgetter(0)(signal_pair)
    # bb = op.itemgetter(1)(signal_pair)
    # tp = tuple(aa, bb)
    tp = zip(*signal_pair)
    tp = tuple(tp)
    return tp


@fts.lru_cache(maxsize=128)
def length(signal_pair: tuple[array]) -> float:
    length_ = map(single.length, signal_pair)
    length_ = tuple(length_)
    return length_


@fts.lru_cache(maxsize=128)
def take(signal_pair: tuple[array]) -> float:
    take_ = map(single.take, signal_pair)
    take_ = tuple(take_)
    return take_


@fts.lru_cache(maxsize=128)
def sampen(signal_pair: tuple[array]) -> float:
    sampen_ = map(single.sampen, signal_pair)
    sampen_ = tuple(sampen_)
    return sampen_


@fts.lru_cache(maxsize=128)
def lag(signal_pair: tuple[array]) -> float:
    lag_ = map(single.lag, signal_pair)
    lag_ = tuple(lag_)
    return lag_


@fts.lru_cache(maxsize=128)
def mav(signal_pair: tuple[array]) -> float:
    mav_ = map(single.mav, signal_pair)
    mav_ = tuple(mav_)
    return mav_


@fts.lru_cache(maxsize=128)
def mad(signal_pair: tuple[array]) -> float:
    mad_ = map(single.mad, signal_pair)
    mad_ = tuple(mad_)
    return mad_


@fts.lru_cache(maxsize=128)
def maxvalue(signal_pair: tuple[array]) -> float:
    maxvalue_ = map(single.maxvalue, signal_pair)
    maxvalue_ = tuple(maxvalue_)
    return maxvalue_


@fts.lru_cache(maxsize=128)
def minvalue(signal_pair: tuple[array]) -> float:
    minvalue_ = map(single.minvalue, signal_pair)
    minvalue_ = tuple(minvalue_)
    return minvalue_


@fts.lru_cache(maxsize=128)
def mean(signal_pair: tuple[array]) -> float:
    meanvalue_ = map(single.mean, signal_pair)
    meanvalue_ = tuple(meanvalue_)
    return meanvalue_


@fts.lru_cache(maxsize=128)
def median(signal_pair: tuple[array]) -> float:
    median_ = map(single.median, signal_pair)
    median_ = tuple(median_)
    return median_


@fts.lru_cache(maxsize=128)
def mode(signal_pair: tuple[array]) -> float:
    mode_ = map(single.mode, signal_pair)
    mode_ = tuple(mode_)
    return mode_


@fts.lru_cache(maxsize=128)
def stdev(signal_pair: tuple[array]) -> float:
    stddev_ = map(single.stdev, signal_pair)
    stddev_ = tuple(stddev_)
    return stddev_


@fts.lru_cache(maxsize=128)
def variance(signal_pair: tuple[array]) -> float:
    variance_ = map(single.variance, signal_pair)
    variance_ = tuple(variance_)
    return variance_


@fts.lru_cache(maxsize=128)
def skew(signal_pair: tuple[array]) -> float:
    skew_ = map(single.skew, signal_pair)
    skew_ = tuple(skew_)
    return skew_


@fts.lru_cache(maxsize=128)
def kurtosis(signal_pair: tuple[array]) -> float:
    kurtosis_ = map(single.kurtosis, signal_pair)
    kurtosis_ = tuple(kurtosis_)
    return kurtosis_


@fts.lru_cache(maxsize=128)
def covariance(signal_pair: tuple[array]) -> float:
    sig_a = op.itemgetter(0)(signal_pair)
    sig_a = tuple(sig_a)
    sig_b = op.itemgetter(1)(signal_pair)
    sig_b = tuple(sig_b)
    cov = map(sts.covariance, sig_a, sig_b)
    cov = tuple(cov)
    return cov
