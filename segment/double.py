#!/bin/env python

import math
import operator as op
import itertools as its
from array import array

import more_itertools as mits

import func_feats.segment.single as segment_i




def synchronized_windows(signals: tuple[array]) -> tuple[tuple[array]]:
    """
    This function multiple signals into syncronized windows of size
    `size` and step `step`
    """
    window_pairs = map(segment_i.window, signals)
    window_pairs = zip(*window_pairs)
    window_pairs = tuple(window_pairs)
    return window_pairs


def synchronized_streams(signals: tuple[array]) -> tuple[tuple[array]]:
    """
    This should do the same thing as teh one above, but for the `whole`
    function
    """
    return (signals,)


def split_continuous(signals: tuple[tuple[array]])\
        -> tuple[tuple[tuple[array]]]:
    """
    input -- iterable-wrap[pair-of_signals[signal]]
    output -- iterable-wrap[pair-of-signals[finite-segments[signal]]]
    """
    segments = map(segment_i.split_continuous, signals)
    segments = zip(*segments)
    segments = tuple(segments)
    acc = []
    for segment in segments:
        foo = map(array, its.repeat('d'), segment)
        foo = tuple(foo)
        acc.append(foo)
    segments = tuple(acc)
    return segments
