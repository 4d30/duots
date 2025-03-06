#!/bin/env python

import operator as op
import itertools as its
import statistics as sts
from array import array

import func_feats.calculators.single as single

def __transpose(signal_pair):
    tp = zip(*signal_pair)
    tp = tuple(tp)
    return tp

def take(signal_pair: tuple[array]) -> float:
    take_ = map(single.take, signal_pair)
    take_ = tuple(take_)
    return take_

#def sampen(signal_pair: tuple[array]) -> float:
#    sampen_ = map(single.sampen, signal_pair)
#    sampen_ = tuple(sampen_)
#    return sampen_


def lag(signal_pair: tuple[array]) -> float:
    lag_ = map(single.lag, signal_pair)
    lag_ = tuple(lag_)
    return lag_


def mav(signal_pair: tuple[array]) -> float:
    mav_ = map(single.mav, signal_pair)
    mav_ = tuple(mav_)
    return mav_


def mad(signal_pair: tuple[array]) -> float:
    mad_ = map(single.mad, signal_pair)
    mad_ = tuple(mad_)
    return mad_


def maxvalue(signal_pair: tuple[array]) -> float:
    maxvalue_ = map(single.maxvalue, signal_pair)
    maxvalue_ = tuple(maxvalue_)
    return maxvalue_


def minvalue(signal_pair: tuple[array]) -> float:
    minvalue_ = map(single.minvalue, signal_pair)
    minvalue_ = tuple(minvalue_)
    return minvalue_


def mean(signal_pair: tuple[array]) -> float:
    meanvalue_ = map(single.mean, signal_pair)
    meanvalue_ = tuple(meanvalue_)
    return meanvalue_


def median(signal_pair: tuple[array]) -> float:
    median_ = map(single.median, signal_pair)
    median_ = tuple(median_)
    return median_


def mode(signal_pair: tuple[array]) -> float:
    mode_ = map(single.mode, signal_pair)
    mode_ = tuple(mode_)
    return mode_


def stdev(signal_pair: tuple[array]) -> float:
    stddev_ = map(single.stdev, signal_pair)
    stddev_ = tuple(stddev_)
    return stddev_


def variance(signal_pair: tuple[array]) -> float:
    variance_ = map(single.variance, signal_pair)
    variance_ = tuple(variance_)
    return variance_


def skew(signal_pair: tuple[array]) -> float:
    skew_ = map(single.skew, signal_pair)
    skew_ = tuple(skew_)
    return skew_

def kurtosis(signal_pair: tuple[array]) -> float:
    kurtosis_ = map(single.kurtosis, signal_pair)
    kurtosis_ = tuple(kurtosis_)
    return kurtosis_

#def kurtosis(signal_pair: tuple[array]) -> float:
#    if isinstance(signal_pair[0], float):
#        kurtosis_ = single.kurtosis(signal_pair)
#    else:
#        kurtosis_ = map(map, its.repeat(single.kurtosis), signal_pair)
#        kurtosis_ = map(array, its.repeat('d'), kurtosis_)
#        kurtosis_ = tuple(kurtosis_)
#    return kurtosis_


# TODO FIX THIS, MAYBE TEST FOR SIGNAL TYPE AND THEN DO THE THING
# -- transform.double.cross_correlation also needs this treatment
def covariance(signal_pair: tuple[array]) -> float:
    sig_a = map(op.itemgetter(0), signal_pair)
    sig_a = tuple(sig_a)

    sig_b = map(op.itemgetter(1), signal_pair)
    sig_b = tuple(sig_b)
    #breakpoint()
    cov = map(sts.covariance, sig_a, sig_b)
    cov = tuple(cov)
    #breakpoint()
    return cov
