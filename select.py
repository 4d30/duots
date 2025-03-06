#!/usr/bin/env python

import csv
import gzip
import itertools as its
import operator as op
import math

from array import array


def valid_pairs(pair: tuple) -> bool:
    """
    tests if the pair of sensors is valid
    """
    if not same_instrument(pair):
        return False

    if not (across_midline(pair)
            or is_hed_cst(pair)):
        return False

    return True


def same_instrument(pair: tuple) -> bool:
    """
    tests if the pair of sensors is valid
    """
    if pair[0]['instrument'] == pair[1]['instrument']:
        return True
    return False


def is_hed_cst(pair: tuple) -> bool:
    """
    tests if the pair of sensors is valid
    """
    if pair[0]['sensors'] == 'hed' \
            and pair[1]['sensors'] == 'cst':
        return True
    return False


def across_midline(pair: tuple) -> bool:
    """
    currently, across the midline is valid
    """
    if pair[0]['sensors'] != 'r':
        return False
    if pair[0].replace('r', 'l') == pair[1]:
        return True
    return False


def valid_two_func(sequence: tuple) -> bool:
    """
    tests if the sequence is valid
    0: segment
    1: transform
    2: calculator
    3: calculator
    """
    names = map(op.itemgetter(0), sequence)
    names = tuple(names)

    # 'split_into_continuous' is not valid here
    if sequence[0][0] == 'split_continuous':
        return False

    # Last step must be take for two func
    if sequence[0][0] == 'synchronized_streams':
        if sequence[3][0] != 'take':
            return False


    # Only one dimension reduction function allowed
    is_xcorr = (sequence[1][0] == 'cross_correlate') 
    is_cov = (sequence[2][0] == 'covariance')
    neither = not(is_xcorr or is_cov)
    if (not op.xor(is_xcorr, is_cov)) or neither:
        return False

    # Doesn't make sense in this context
    if is_cov:
        if sequence[1][0] == 'findpeaks':
            return False
    return True


#def valid_three_func(sequence: tuple) -> bool:
#    """
#    tests if the sequence is valid
#    0: segment
#    1: transform
#    2: calculator
#    3: calculator
#    4: [symm, avg, ...]
#    """
#    # 'split_into_continuous' is not valid here
#    if sequence[0][0] == 'split_continuous':
#        return False
#
#    if sequence[0][0] != 'synchronized_windows':
#        return False
#
#    is_xcorr = (sequence[1][0] == 'cross_correlate') 
#    cov1 = (sequence[2][0] == 'covariance') 
#    if cov1:
#        return False
#    cov2 = (sequence[3][0] == 'covariance') 
#    neither = not(is_xcorr or cov2)
#    if (not op.xor(is_xcorr, cov2)) or neither:
#        return False
#    return True

def valid_four(sequence: tuple) -> bool:
    """
    tests if the sequence is valid

    four function sequences will calculate symmetry and average across
    midline

    0: segment
    1: transform
    2: calculator
    3: transpose
    4: calculator
    5: [symm, avg, ...]
    """
    if len(sequence) != 6:
        return False
    # 'split_continuous' is not valid here
    if sequence[0][0] == 'split_continuous':
        return False

    if sequence[0][0] == 'synchronized_streams'\
        and sequence[4][0] != 'take':
        return False

    if sequence[0][0] == 'synchronized_windows'\
        and sequence[4][0] == 'take':
        return False

    if sequence[1][0] == 'cross_correlate':
        return False

    if sequence[2][0] == 'take':
        return False

    if sequence[2][0] == 'covariance':
        return False

    if sequence[3][0] == 'covariance':
        return False

    if sequence[3][0] == 'take':
        return False

    if sequence[4][0] == 'covariance':
        return False


#    if sequence[4][0] == 'take':
#        return False

    if sequence[5][0] == 'take':
        return False

    # Windowing is required for DFT
    if sequence[1][0] == 'dft' and sequence[0][0] != 'synchronized_windows':
        return False
    return True


def _behavior(data: tuple[dict], params: tuple) -> tuple[bool]:
    """
    creates mask to indicate where the event is
    params:
    0: sensor
    1: instrument
    2: event
    3: behavior
    """
    if params['behavior'] is None:
        return tuple(its.repeat(True, len(data)))
    is_behavior = map(op.itemgetter('behavior'), data)
    is_behavior = map(float, is_behavior)
    is_behavior = map(int, is_behavior)
    is_behavior = map(op.eq, is_behavior, its.repeat(params['behavior_id']))
    is_behavior = tuple(is_behavior)
    is_behavior = tuple(is_behavior)
    return is_behavior


def _event(data: tuple[dict], params: tuple) -> tuple[bool]:
    """
    creates mask to indicate where the event is
    params:
    0: sensor
    1: instrument
    2: event
    3: behavior
    """
    is_event = map(op.itemgetter('event'), data)
    is_event = map(float, is_event)
    is_event = map(int, is_event)
    is_event = map(op.eq, is_event, its.repeat(params['event_id']))
    is_event = tuple(is_event)
    return is_event


def _instrument(data: tuple[dict], params: tuple) -> tuple[bool]:
    """
    creates mask to indicate where the event is
    params:
    0: sensor
    1: instrument
    2: event
    3: behavior
    """
    is_instrument = tuple(data[0].keys())
    is_instrument = map(op.contains,
                        is_instrument,
                        its.repeat(params['instrument']))
    is_instrument = tuple(is_instrument)
    return is_instrument


def _sensor(data: tuple[dict], params: tuple) -> tuple[bool]:
    is_sensor = tuple(data[0].keys())
    is_sensor = map(op.contains, is_sensor, its.repeat(params['sensor']))
    is_sensor = tuple(is_sensor)
    return is_sensor


def datafile(session: dict) -> tuple[dict]:
    """
    returns a tuple of dict

    session:
    """
    data = session['path']
    handle = gzip.open(data, 'rt')
    data = csv.DictReader(handle)
    data = tuple(data)
    handle.close()
    return data


def _copy_params(params: dict) -> tuple[dict]:
    sensor_params = params['sensors']
    keys = ('group_id', 'sensor',
            'instrument', 'instrument_id',
            'event', 'event_id',
            'behavior', 'behavior_id')
    # make a copy of params for each sensor in params['sensors']
    exploded_params = zip(its.repeat(params['group_id']),
                          sensor_params,
                          its.repeat(params['instrument']),
                          its.repeat(params['instrument_id']),
                          its.repeat(params['event']),
                          its.repeat(params['event_id']),
                          its.repeat(params['behavior']),
                          its.repeat(params['behavior_id']))
    exploded_params = map(zip, its.repeat(keys), exploded_params)
    exploded_params = map(dict, exploded_params)
    exploded_params = tuple(exploded_params)
    return exploded_params


def _align_streams(streams: tuple) -> tuple:
    # mask off regions where any stream is not finite
    is_finite = zip(*streams)
    is_finite = map(lambda x: all(map(math.isfinite, x)), is_finite)
    is_finite = tuple(is_finite)
    acc = []
    for a_stream in streams:
        masked = its.starmap(lambda f, b: f if b else float('nan'),
                             zip(a_stream, is_finite))
        masked = array('d', masked)
        acc.append(masked)
    these_streams = tuple(acc)
    return these_streams


def streams(data: tuple[dict], params: tuple) -> tuple[array]:
    """
    selects the streams from the data
    params:
    0: sensors
    1: instrument
    2: event
    3: behavior
    """

    # explode params
    copied_params = _copy_params(params)
    # reuse the stream function to load each stream
    these_streams = map(stream,
                        its.repeat(data),
                        copied_params)
    these_streams = tuple(these_streams)
    # all streams present?
    any_data = map(bool, these_streams)
    any_data = all(any_data)
    if not any_data:
        return None

    # mask off regions where any stream is not finite
    these_streams = _align_streams(these_streams)

    # all streams finite?
    any_data = map(math.isfinite, these_streams[0])
    any_data = any(any_data)
    if not any_data:
        return None
    return these_streams


def stream(data: tuple[dict], params: tuple) -> array:
    """
    selects a stream from the data and masks off where the event is
    params:
    0: sensor
    1: instrument
    2: event
    3: behavior
    """
    keys = tuple(data[0].keys())
    this_sensor = _sensor(data, params)
    this_instrument = _instrument(data, params)
    this_stream = map(op.and_, this_sensor, this_instrument)
    this_stream = next(its.compress(keys, this_stream), None)
    if not this_stream:
        return None
    this_stream = map(op.itemgetter(this_stream), data)
    this_stream = map(float, this_stream)

    is_event = _event(data, params)
    is_behavior = _behavior(data, params)
    these_segments = map(op.and_, is_event, is_behavior)
    these_segments = zip(this_stream, these_segments)
    these_segments = its.starmap(lambda f, b: f if b else float('nan'),
                                 these_segments)
    these_segments = array('f', these_segments)
    if not these_segments:
        return None

    is_finite = map(op.ne, these_segments, these_segments)
    is_finite = tuple(is_finite)
    if not any(is_finite):
        return None
    return these_segments
