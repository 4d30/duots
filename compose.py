#!/usr/bin/env python

import functools as fts

from typing import Callable


def apply(val: float, func: Callable) -> float:
    """Apply a function to a value.
    kwargs:
    val -- a value
    func -- a callable function
    """
    return func(val)


def functions(funcs: tuple[Callable]) -> Callable:
    """Compose a list of functions.
    kwargs:
    funcs -- a tuple of functions
    """
    composed = fts.partial(fts.reduce, apply, funcs)
    return composed


def descriptors(descs: tuple[str]) -> str:
    """Compose a list of descriptors.
    kwargs:
    descs -- a tuple of descriptive strings
    """
    descs = filter(lambda x: x != 'None', descs)
    descs = '_'.join(descs)
    return descs

