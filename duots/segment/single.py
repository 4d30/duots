#!/bin/env python

import math
import operator as op
import itertools as its
from array import array

import more_itertools as mits


def window(signal: array,
           size: int = 200,
           step: int = 25) -> iter:
    """
    This function segments a signal into
    windows of size `size` and step `step`
    """
    nan = float('nan')
    win = mits.windowed(signal, size, fillvalue=nan, step=step)

    def all_finite(x):
        return all(map(math.isfinite, x))

    win = filter(all_finite, win)
    #win = map(tuple, win)
    win = tuple(win)
    return win


def whole(signal: array) -> array:
    return (tuple(signal),)


def split_continuous(signal: array) -> tuple[array]:
    """
    This function splits a signal into
    continuous segments
    """
    def begins(a, b):
        a_is_nan = op.ne(a, a)
        b_is_not_nan = op.not_(op.ne(b, b))
        return op.and_(a_is_nan, b_is_not_nan)

    def ends(a, b):
        a_is_not_nan = op.not_(op.ne(a, a))
        b_is_nan = op.ne(b, b)
        return op.and_(a_is_not_nan, b_is_nan)

    segments = mits.split_when(signal, begins)
    segments = filter(lambda x: any(map(math.isfinite, x)), segments)
    segments = map(mits.split_when, segments, its.repeat(ends))
    segments = its.chain.from_iterable(segments)
    segments = filter(lambda x: all(map(math.isfinite, x)), segments)
    segments = tuple(segments)
    return segments
