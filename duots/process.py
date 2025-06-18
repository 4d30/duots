#!/usr/bin/env

import operator as op
import func_feats.compose as compose


def get_funcs(process):
    funcs = map(op.itemgetter(1), process)
    funcs = tuple(funcs)
    return funcs

def get_names(process):
    names = map(op.itemgetter(0), process)
    names = tuple(names)
    return names


def average(process, segments):
    names = map(op.itemgetter(0), process)
    names = tuple(names)
    funcs = map(op.itemgetter(1), process)
    funcs = tuple(funcs)
    proc = compose.functions(funcs)
    desc = '__'.join(names)
    weights = map(len, segments)
    weights = tuple(weights)

    values = map(proc, segments)
    values = tuple(values)
    values = map(op.mul, values, weights)
    values = sum(values)
    values = op.truediv(values, sum(weights))
    return desc, values


def name(process):
    names = map(op.itemgetter(0), process)
    names = tuple(names)
    name = compose.descriptors(names)
    return name
