#!/bin/env python

import math
import operator as op
import itertools as its
import functools as fts
from array import array

import more_itertools as mits

from segment import single as segment_i


@fts.lru_cache(maxsize=128)
def synchronized_windows(signals: tuple[array]) -> tuple[tuple[array]]:
    """
    This function multiple signals into syncronized windows of size
    `size` and step `step`
    (2xN) -> (2xNwxLw)
    """
    window_pairs = map(segment_i.window, signals)
    # window_pairs = zip(*window_pairs)
    window_pairs = tuple(window_pairs)
    return window_pairs


@fts.lru_cache(maxsize=128)
def synchronized_streams(signals: tuple[array]) -> tuple[tuple[array]]:
    """
    This should do the same thing as teh one above, but for the `whole`
    function
    (2xN) -> (2x1xN)
    """
    signals = ((signals[0],), (signals[1],),)
    return signals


@fts.lru_cache(maxsize=128)
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
        foo = map(tuple, segment)
        foo = tuple(foo)
        acc.append(foo)
    segments = tuple(acc)
    return segments
