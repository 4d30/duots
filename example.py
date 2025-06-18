#!/usr/bin/env python

from duots import generate, compose


def main():
    import math
    import operator as op
    import itertools as its

    sig_a = range(1, 1000)
    sig_a = reversed(sig_a)
    sig_a = map(op.truediv, its.repeat(4*math.pi), sig_a)
    sig_a = map(math.sin, sig_a)
    sig_a = tuple(sig_a)

    sig_b = range(1, 1000)
    sig_b = reversed(sig_b)
    sig_b = map(op.truediv, its.repeat(4*math.pi), sig_b)
    sig_b = map(math.cos, sig_b)
    sig_b = tuple(sig_b)

    s0 = (sig_a, sig_b,)
    ii = 0
    for proc in generate.processes():
        names, funcs = zip(*proc)
        name = '__'.join(names)
        funcs = tuple(funcs)

        ff = compose.functions(funcs)
        value = ff(s0)
        print(name, value)


if __name__ == "__main__":
    main()
