#!/usr/bin/env python

import inspect as ins
import itertools as its
import functools as fts

import types

from . import filter_select
from .segment import double as segment_ii
from .transform import double as transform_ii
from .calculators import single as calculator_i
from .calculators import double as calculator_ii
from .calculators import features as features


def _make_generator(module: types.ModuleType) -> types.GeneratorType:
    iterable = ins.getmembers(module, ins.isroutine)
    for name, func in iterable:
        if not name.startswith('__'):
            yield name, func


def processes():
    procs = its.chain(four_func(),)  # three_func(), four_func())
    procs = its.chain(two_func(),)  # four_func())
    procs = its.chain(four_func(), two_func(),)  # four_func())
    yield from procs


def two_func():
    # Cross-correlation & covariance
    ss = fts.partial(_make_generator, segment_ii)()
    tt = fts.partial(_make_generator, transform_ii)()
    c1 = fts.partial(_make_generator, calculator_ii)()
    c2 = fts.partial(_make_generator, calculator_i)()
    iterable = its.product(ss, tt, c1, c2)
    iterable = filter(filter_select.valid_two_func, iterable)
    yield from iterable


def three_func():
    # I don't think three-func is useful
    ss = fts.partial(_make_generator, segment_ii)()
    ss = filter(lambda x: x[0] == 'synchronized_windows', ss)
    tt = fts.partial(_make_generator, transform_ii)()  # xcorr only
    c1 = fts.partial(_make_generator, calculator_ii)()
    c2 = fts.partial(_make_generator, calculator_ii)()
    c3 = fts.partial(_make_generator, calculator_ii)()
    iterable = its.product(ss, tt, c1, c2, c3)
    iterable = filter(filter_select.valid_three_func, iterable)
    yield from iterable


def four_func():
    # Symmetry & avg
    ss = fts.partial(_make_generator, segment_ii)()
    tt = fts.partial(_make_generator, transform_ii)()
    c1 = fts.partial(_make_generator, calculator_ii)()
    # c2 = (('transpose', calculator_ii.__transpose,),)
    c3 = fts.partial(_make_generator, calculator_ii)()
    c4 = fts.partial(_make_generator, features)()
    iterable = its.product(ss, tt, c1, c3, c4)
    iterable = filter(filter_select.valid_four, iterable)
    yield from iterable
