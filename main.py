#!/usr/bin/env python

import time
import operator as op
import itertools as its
from array import array
import multiprocessing as mp
import collections as cts
import io
import csv
import configparser
import warnings

import func_feats.psql as psql
import func_feats.generate as generate
import func_feats.select as select
import func_feats.segment.double as segment
import func_feats.calculators.single as c1
import func_feats.calculators.double as c2
import func_feats.compose as compose

import corbett_db.exception_log as elog


from numba.core.errors import NumbaPerformanceWarning


def load_processnames(config):
    QQ = """ SELECT
            feature_name,
            process_id
         from feature.process_names"""
    data = psql.execute(config, QQ)
    names = map(op.itemgetter('feature_name'), data)
    ids = map(op.itemgetter('process_id'), data)
    data = zip(names, ids)
    data = dict(data)
    return data


def compute(session):
    config = session['config']
    warnings.simplefilter("ignore", NumbaPerformanceWarning)
    #time.sleep(session['spl_row']*3)
    t0 = time.time()
    fieldnames = ('spl_row',
                  'event_id',
                  'behavior_id',
                  'group_id',
                  'instrument_id',
                  'process_id',
                  'value')
    t0 = time.time()
    data = select.datafile(session)
    pids = load_processnames(config)
    #data = session['data']
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    for params in generate.params(config):
        streams = select.streams(data, params)
        if not streams:
            continue
        segments = segment.split_continuous(streams)
        segments = filter(lambda x: len(x[0]) > 200, segments)
        segments = tuple(segments)
        if not segments:
            continue
        for proc in generate.processes():
            try:
                names = map(op.itemgetter(0), proc)
                names = tuple(names)
                name_key = '__'.join(names)
                pid = pids[name_key]
                #pid = psql.read_process_table(config, names)
                funcs = map(op.itemgetter(1), proc)
                funcs = tuple(funcs)
                #f1 = compose.functions(funcs[:1])
                #f2 = compose.functions(funcs[:2])
                #f3 = compose.functions(funcs[:3])
                #f4 = compose.functions(funcs[:4])
                #f5 = compose.functions(funcs[:5])
                #s0 = segments[0]
                ff = compose.functions(funcs)

                #print(names)
                #breakpoint()
                vals = map(ff, segments)
                ss = tuple(vals)
                wts = map(len, segments)
                wts = array('d', wts)
                val = map(op.mul, ss, wts)
                val = sum(val)
                val = op.truediv(val, sum(wts))
                if val != val:
                    continue
#                print(session['spl_row'],
#                      '__'.join(names),
#                      val)
#                print()
                writer.writerow({'spl_row': session['spl_row'],
                                 'event_id': params['event_id'],
                                 'behavior_id': params['behavior_id'],
                                 'group_id': params['group_id'],
                                 'instrument_id': params['instrument_id'],
                                 'process_id': pid,
                                 'value': val})

            except Exception as e:
                print(session['spl_row'], e)
                print('__'.join(names))
                elog.log_exception(config, 'func_feats/main.py',
                                   identifier_type='spl_row',
                                   identifier_value=session['spl_row'])
        buf.seek(0)
        psql.write_session(config, buf)
        buf.truncate(0)
        buf.seek(0)
       
    buf.close()
    t1 = time.time()
    print('session', session['spl_row'], 'delta_t', t1-t0, 's')
    return None

def calculate_chunksize(config):
    n_tasks = generate.sessions(config)
    n_tasks = tuple(n_tasks)
    n_tasks = len(n_tasks)
    n_workers = mp.cpu_count()
    chunksize = max(1, n_tasks // (n_workers *2))
    return chunksize

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
#    for proc in generate.processes():
#        names = map(op.itemgetter(0), proc)
#        names = tuple(names)
#        psql.load_process(config, names)

    chunksize = calculate_chunksize(config)
    sessions = generate.sessions(config)

#    proc = its.starmap(compute, iterable,)
#    cts.deque(proc, maxlen=0)
#    breakpoint()
#    return None

    pool = mp.Pool(mp.cpu_count())
    pool.map(compute, sessions, chunksize=chunksize)
    pool.close()
    pool.join()
    return None


if __name__ == '__main__':
    main()
