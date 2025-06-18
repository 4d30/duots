#!/usr/bin/env python

import os
import inspect as ins
import itertools as its
import functools as fts
import collections as cts

import types

import duots.filter_select as filter_select
import duots.segment.double as segment_ii
import duots.transform.double as transform_ii
import duots.calculators.single as calculator_i
import duots.calculators.double as calculator_ii
import duots.calculators.features as features

import duots.psql as psql


def sensor_groups(config):
    QQ = """
        SET search_path TO feature;
        SELECT sg.id AS group_id,
            array_agg(s.name) AS sensors
        FROM sensor_group sg
        JOIN sensor_group_members sgm
            ON sg.id = sgm.group_id
        JOIN sensor s
            ON sgm.sensor_id = s.id
        GROUP BY sg.id
        ORDER BY sg.id;
        """
    ss = psql.execute(config, QQ)
    yield from ss


def instruments(config):
    QQ = """
        SET search_path TO feature;
        SELECT ii.id AS instrument_id,
        ii.name AS instrument
        FROM instrument ii
        ORDER BY ii.id;
        """
    ii = psql.execute(config, QQ)
    yield from ii


def events(config):
    QQ = """
        SET search_path TO feature;
        SELECT e.id AS event_id,
        e.name AS event
        FROM event e
        ORDER BY e.id;
        """
    ee = psql.execute(config, QQ)
    yield from ee


def behaviors(config):
    QQ = """
        SET search_path TO feature;
        SELECT b.id AS behavior_id,
        b.name AS behavior
        FROM behavior b
        ORDER BY b.id;
        """
    bb = psql.execute(config, QQ)
    bb = its.chain([{'behavior_id': 'nan', 'behavior': None}], bb)
    bb = tuple(bb)
    yield from bb


def _make_generator(module: types.ModuleType) -> types.GeneratorType:
    iterable = ins.getmembers(module, ins.isroutine)
    for name, func in iterable:
        if not name.startswith('__'):
            yield name, func


def processes():
    procs = its.chain(four_func(),)# three_func(), four_func())
    procs = its.chain(two_func(),)# four_func())
    procs = its.chain(four_func(), two_func(),)# four_func())
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
    tt = fts.partial(_make_generator, transform_ii)() # xcorr only
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


def sessions(config, session=None):
    if session:
        QQ = """SELECT distinct spl.spl_row,
                combined_file as path
             FROM spl
             JOIN process_files pf
                 ON spl.spl_row = pf.spl_row
             LEFT JOIN feature.values v
                 ON spl.spl_row = v.spl_row
             WHERE combined_file is not null
                 AND v.spl_row = {}
             ORDER BY spl_row;""".format(session)
    else:
        QQ = """SELECT distinct spl.spl_row,
                combined_file as path
             FROM spl
             JOIN process_files pf
                 ON spl.spl_row = pf.spl_row
             LEFT JOIN feature.values v
                 ON spl.spl_row = v.spl_row
             WHERE combined_file is not null
                 AND v.spl_row is null
             ORDER BY spl_row;"""
    sessions = psql.execute(config, QQ)
    rto = config['paths']['rto']
    # rto = '/mnt/data/corbett/'
    for session in sessions:
        session['config'] = config
        session['path'] = os.path.join(rto, session['path'])
        yield session


def params(config):
    ss = sensor_groups(config)
    ii = instruments(config)
    ee = events(config)
    bb = behaviors(config)
    iterable = its.product(ss, ii, ee, bb)
    for each in iterable:
        yield cts.ChainMap(*each)
