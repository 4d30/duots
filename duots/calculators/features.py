#!/bin/env python

import operator as op
import itertools as its


def symmidx(value_pair: tuple[float]) -> float:
    val_a, val_b = value_pair
    if val_a == val_b:
        return 100

    abs_a = op.abs(val_a)
    abs_b = op.abs(val_b)

    if abs_a <= abs_b:
        num = abs_a
        den = abs_b
    else:
        num = abs_b
        den = abs_a
    if (den == 0):
        return float('nan')

    sign_swap = (val_a, val_b,)
    sign_swap = map(op.gt, sign_swap, its.repeat(0))
    if op.xor(*sign_swap):
        num = op.mul(num, -1)

    symm = op.truediv(num, den)
    symm = op.mul(symm, 100)
    return symm


def avg(value_pair: tuple[float]) -> float:
    value = sum(value_pair)
    value = value/len(value_pair)
    return value
