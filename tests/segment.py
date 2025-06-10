#!/usr/bin/env python


import os
import csv
import math
import unittest
import configparser
import operator as op
import collections as cts
import itertools as its
import random
from array import array # noqa
import statistics as sts

import matplotlib.pyplot as plt


import func_feats.psql as psql
import func_feats.generate as generate
import func_feats.select as select
import func_feats.segment.double as segment_ii
import func_feats.transform.double as transform_ii

import matplotlib.pyplot as plt # noqa


class TestStreamTransformation(unittest.TestCase):

    def make_streams(self):

        self.duration = 2.0*math.pi
        self.freq = 10
        self.A = 1.0
        self.length = int(100*self.duration)

        t = range(0, self.length)
        dt = self.duration / self.length
        t = map(op.mul, its.repeat(dt), t)
        self.t = tuple(t)

        self.omega = 2.0 * math.pi * self.freq

        aa = map(op.mul, its.repeat(self.omega), self.t)
        aa = map(math.sin, aa)
        aa = map(op.mul, aa, its.repeat(self.A))
        aa = list(aa)

        bb = map(op.mul, its.repeat(self.omega), self.t)
        bb = map(math.cos, bb)
        bb = map(op.mul, bb, its.repeat(self.A))
        bb = list(bb)

        self.n_breaks = random.randint(1, 5)
        last = 0
        for one_break in range(self.n_breaks):
            start = random.randint(last, self.length)
            thislength = random.randint(0, (self.length-start)//2)
            aa[start:start+thislength] = [float('nan')]*thislength
            bb[start:start+thislength] = [float('nan')]*thislength
            last = start+thislength
        self.streams = (tuple(aa), tuple(bb),)

    def setUp(self):
        self.make_streams()
        pass

    def test_passalong(self):
        segments_a = segment_ii.split_continuous(self.streams)
        segments_b = segment_ii.split_continuous(self.streams)
        for pair in zip(segments_a, segments_b):
            test = transform_ii.passalong(pair[0])
            test = map(op.eq, test, pair[1])
            self.assertTrue(all(test))

    def test_dft(self):
        segments = segment_ii.split_continuous(self.streams)
        for segment in segments:
            windows = segment_ii.synchronized_windows(segment)
            dfts = transform_ii.dft(windows)
            for spectrum in dfts:
                max_amp = max(spectrum[0])
                # Expected value is (A * N)/2
                # where N is the number of samples
                # and A is the amplitude
                # 0.5 is power correction for Hann window
                expected = (self.A * len(windows[0][0])/2) * 0.5
                self.assertAlmostEqual(max_amp, expected, delta=0.5)

    def test_autocorrelate(self):
        segments = segment_ii.split_continuous(self.streams)
        for segment in segments:
            windows = segment_ii.synchronized_windows(segment)
            products = transform_ii.autocorrelate(windows)
            for product in products:
                # Check that max amplitude is equal to expectation
                # value for the test signal
                #
                # We expect sine to eval to 1/2
                #       T
                # 1/T *  ∫ sin^2 (ωt) dt = 1/2
                #       0

                corr = product[0]
                max_amp = max(corr)
                expected = map(op.mul, its.repeat(self.omega), self.t)
                expected = map(math.sin, expected)
                expected = map(pow, expected, its.repeat(2))
                expected = sum(expected)
                expected = expected / len(self.t)
                expected = expected * len(windows[0][0])
                self.assertAlmostEqual(max_amp, expected, delta=0.5)

                # Check that the correlation of test signal
                # at lag > 0 is within [-100, 100]
                for i in range(1, len(corr)):
                    self.assertTrue(-100 <= corr[i] <= 100,)

    def test_zerosq(self):
        segments = segment_ii.split_continuous(self.streams)
        for segment in segments:
            windows = segment_ii.synchronized_windows(segment)
            products = transform_ii.zerosq(windows)
            for product in products:
                max_amp = max(product[0])
                median = sts.median(windows[0][0])
                expected = map(op.sub, windows[0][0], its.repeat(median))
                expected = map(pow, expected, its.repeat(2))
                expected = max(expected)
                self.assertAlmostEqual(max_amp, expected, delta=0.05)
                all_gt_zero = all(map(op.gt, product[0], its.repeat(0)))
                self.assertTrue(all_gt_zero)

    def test_findpeaks(self):
        segments = segment_ii.split_continuous(self.streams)
        for segment in segments:
            windows = segment_ii.synchronized_windows(segment)
            peaksets = transform_ii.findpeaks(windows)
            for peakset in peaksets:
                expected = len(windows[0][0])/(self.freq * 1.0)
                self.assertAlmostEqual(len(peakset[0]), expected, delta=1.0)
                for peak in peakset[0]:
                    self.assertAlmostEqual(peak, self.A, delta=0.05)
        pass


class TestStreamSegmentation(unittest.TestCase):

    def make_streams(self):
        length = 1000000
        aa = [1.0]*length
        bb = [1.0]*length
        self.n_breaks = random.randint(0, 5)
        last = 0
        for one_break in range(self.n_breaks):
            start = random.randint(last, length)
            thislength = random.randint(0, (length-start)//2)
            aa[start:start+thislength] = [float('nan')]*thislength
            bb[start:start+thislength] = [float('nan')]*thislength
            last = start+thislength
        self.streams = (tuple(aa), tuple(bb),)

    def setUp(self):
        self.make_streams()
        pass

    def test_synchronized_windows(self):
        segments = segment_ii.split_continuous(self.streams)
        n_segments = len(segments)
        self.assertEqual(n_segments, self.n_breaks+1)
        for segment in segments:
            windows = segment_ii.synchronized_windows(segment)
            n_windows = len(windows[0])
            # window width = 200, step = 25
            expected = math.floor((len(segment[0]) - 200)/25) + 1
            self.assertEqual(n_windows, expected)

    def test_synchronized_streams(self):
        segments = segment_ii.split_continuous(self.streams)
        for segment in segments:
            streams = segment_ii.synchronized_streams(segment)
            self.assertEqual(len(streams[0][0]), len(segment[0]))


class TestStreamSelection(unittest.TestCase):

    def fetch_ecf(self, config, spl_row):
        QQ = """SELECT ecf_path
        FROM session_paths_view spv
        WHERE spv.spl_row = {};""".format(spl_row)
        ecf = psql.execute(config, QQ)
        ecf = ecf[0]['ecf_path']
        ecf = os.path.join(config['paths']['rto'], ecf)
        handle = open(ecf, 'r')
        ecf = csv.DictReader(handle)
        ecf = tuple(ecf)
        handle.close()
        return ecf

    def fetch_placed_sensors(self, config, spl_row):
        QQ = """SELECT sensor_loc
        FROM spl
        WHERE spl_row = {};""".format(spl_row)
        placed_sensors = psql.execute(config, QQ)
        placed_sensors = map(op.itemgetter('sensor_loc'), placed_sensors)
        placed_sensors = tuple(placed_sensors)
        return placed_sensors

    def setUp(self):
        self.spl_row = 318  # CT072 3M
        self.spl_row = 352  # DB080 3M
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.config = config
        sessions = generate.sessions(config, session=self.spl_row)
        self.session = next(sessions, None)
        self.datafile = select.datafile(self.session)
        self.ecf = self.fetch_ecf(config, self.spl_row)
        self.events = cts.Counter()
        self.events.update(map(op.itemgetter('Event'), self.ecf))
        self.placed_sensors = self.fetch_placed_sensors(config, self.spl_row)
        self.t_start = next(filter(lambda x: x['Event'] != '8'
                                   and x['Event'] != '7',
                                   self.ecf))
        self.t_start = float(self.t_start['Sensor Time(ms)'])
        self.t_end = its.pairwise(reversed(self.ecf))
        self.t_end = filter(lambda x: x[0]['Event'] == '8'
                            and (x[1]['Event'] != '7'
                                 and x[1]['Event'] != '8'), self.t_end)
        self.t_end = float(next(self.t_end)[0]['Sensor Time(ms)'])
        qq = """ SELECT sensor_location,
        event
        from event_trial_drop_view etdv
        JOIN spl_view sv
        ON etdv.subject_id = sv.subject_id
        AND etdv.timepoint = sv.timepoint
        WHERE sv.spl_row = {}
        AND fall_reason = 'drop'
        ;""".format(self.spl_row)
        self.drops = psql.execute(self.config, qq)

    def test_select_streams(self):
        # test if streams are selected vs placed in spl
        for params in filter(lambda x: x['behavior'] is None,
                             generate.params(self.config)):
            streams = select.streams(self.datafile, params)
            if not streams:
                continue
            for sensor, stream in zip(params['sensors'], streams):
                QQ = """ SELECT {}
                FROM session_paths_view spv
                WHERE spv.spl_row = {};""".format(sensor, self.spl_row)
                raw_data = psql.execute(self.config, QQ)
                raw_data = raw_data[0][sensor]
                raw_data = os.path.join(self.config['paths']['rto'], raw_data)
                handle = open(raw_data, 'r')
                raw_data = csv.DictReader(handle)
                raw_data = map(op.itemgetter('Time(ms)'), raw_data)
                raw_data = tuple(raw_data)
                handle.close()
                if sensor in self.placed_sensors:
                    self.assertTrue(any(map(math.isfinite, stream)))

    def test_split_continuous(self):
        for params in filter(lambda x: x['behavior'] is None,
                             generate.params(self.config)):
            streams = select.streams(self.datafile, params)
            if not streams:
                continue
            segments = segment_ii.split_continuous(streams)
            n_segments = len(segments)
            n_events = filter(lambda x: int(x['Event']) == params['event_id'],
                              self.ecf)
            n_events = len(tuple(n_events))
            n_drops = filter(lambda x: (int(x['event']) == params['event_id'])
                             and (x['sensor_location'] in params['sensors']),
                             self.drops)
            n_drops = len(tuple(n_drops))
            self.assertEqual(n_segments, (n_events + n_drops))


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestStreamSegmentation))
    # suite.addTest(loader.loadTestsFromTestCase(TestStreamTransformation))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    # unittest.main(verbosity=2)
