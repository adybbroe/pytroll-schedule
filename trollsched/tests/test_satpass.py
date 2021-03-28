#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 - 2021 Pytroll developers

# Author(s):

#   Adam.Dybbroe <adam.dybbroe@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Test the satellite pass and swath boundary classes
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from trollsched.satpass import Pass
from trollsched.boundary import SwathBoundary
from pyorbital.orbital import Orbital
from pyresample.geometry import AreaDefinition, create_area_def

LONS1 = np.array([-122.29913729160562, -131.54385362589042, -155.788034272281,
                  143.1730880418349, 105.69172088208997, 93.03135571771092,
                  87.26010432019743, 83.98598584966442, 81.86683434871546,
                  80.37175346216411, 79.2509798668123, 78.37198926578984,
                  77.65800714027662, 77.06147400915819, 76.55132566889495,
                  76.10637628220547, 75.71164306799828, 75.35619180525052,
                  75.03181505238287, 74.73218847041143, 74.45231256197947,
                  74.18813012461848, 73.9362540393912, 73.69376447765231,
                  73.45804804675883, 73.22665809422263, 72.99717692793544,
                  72.76705638792168, 72.53339841609603, 72.2925978414254,
                  72.03965795937306, 71.76661774368146, 71.45957469190316,
                  57.97687872167697, 45.49802548616658, 34.788857347919546,
                  25.993525469714424, 18.88846123000295, 13.14317179269443,
                  8.450362684274728, -0.27010733525252295, -3.0648326302431794,
                  -5.116189000358824, -6.73429807721795, -8.072680053386163,
                  -9.21696007364773, -10.220171884036919, -11.11762132513045,
                  -11.934120125548072, -12.687881125682765, -13.392781001351315,
                  -14.059756511026736, -14.69771138916782, -15.314133712696703,
                  -15.915536409615735, -16.507788289068856, -17.09637839792269,
                  -17.686643087306685, -18.283978247944123, -18.894056410060063,
                  -19.523069195727878, -20.17801994245519, -20.867100607022966,
                  -21.600204055760642, -22.389653641849733, -23.251288693929943,
                  -24.206153922914886, -25.283264445138713, -26.524411381004743,
                  -27.993172418988525, -29.79361072725673, -32.11515837055801,
                  -35.36860848223405, -35.38196057933595, -35.96564490844792,
                  -37.14469461070555, -39.34032289002443, -43.49756191648018,
                  -52.140150361811244, -73.32968630186114], dtype='float64')

LONS1 = np.array([-122.29913729, -131.54385363, -155.78803427, 143.17308804,
                  105.69172088, 93.03135572, 87.26010432, 83.98598585,
                  81.86683435, 80.37175346,   79.25097987,   78.37198927,
                  77.65800714, 77.06147401,   76.55132567,   76.10637628,
                  75.71164307, 75.35619181,   75.03181505,   74.73218847,
                  74.45231256, 74.18813012,   73.93625404,   73.69376448,
                  73.45804805, 73.22665809,   72.99717693,   72.76705639,
                  72.53339842, 72.29259784,   72.03965796,   71.76661774,
                  71.45957469, 71.39305192,   71.32650197,   71.25992437,
                  71.19331862, 71.12668423,   57.93375961,   57.8691876,
                  57.80474299, 57.74042519,   57.67623356,   45.47889659,
                  45.42167224, 45.36469063,   45.30795076,   45.25145168,
                  34.78885735, 34.74114338,   34.69373177,   34.64662123,
                  34.59981051, 34.55329839,   25.96780733,   25.92949682,
                  25.89150514, 25.85383096,   25.81647298,   18.8781428,
                  18.84739757, 18.81696643,   18.78684811,   18.75704136,
                  13.14317179, 13.1183538,   13.09383559,   13.06961602,
                  13.04569396, 13.02206833,    8.43688019,    8.41689195,
                  8.39718542,  8.37775961,    8.35861358,    4.5581567,
                  4.5416334,   4.52537524,    4.50938138,   -0.27010734,
                  -3.06483263, -5.116189,   -6.73429808,   -8.07268005,
                  -9.21696007, -10.22017188,  -11.11762133,  -11.93412013,
                  -12.68788113, -13.392781,  -14.05975651,  -14.69771139,
                  -15.31413371, -15.91553641,  -16.50778829,  -17.0963784,
                  -17.68664309, -18.28397825,  -18.89405641,  -19.5230692,
                  -20.17801994, -20.86710061,  -21.60020406,  -22.38965364,
                  -23.25128869, -24.20615392,  -25.28326445,  -26.52441138,
                  -27.99317242, -29.79361073,  -32.11515837,  -35.36860848,
                  -35.18255168, -35.18465605,  -35.18708062,  -35.18982603,
                  -35.19289295, -35.3841169,  -35.38765191,  -35.39154828,
                  -35.39580684, -35.40042851,  -35.96745106,  -35.97314753,
                  -35.97926188, -35.98579532,  -35.99274907,  -37.14469461,
                  -37.15393687, -37.16367838,  -37.17392086,  -37.18466611,
                  -37.19591597, -39.3508544,  -39.36717234,  -39.38411749,
                  -39.40169261, -39.41990053,  -43.50738307,  -43.53740579,
                  -43.56827095, -43.5999832,  -43.63254727,  -52.14015036,
                  -52.20477195, -52.2706551,  -52.3378081,  -52.40623933,
                  -52.47595734, -73.44197755,  -73.61201994,  -73.78399304,
                  -73.95790125, -74.13374872, -120.99699021, -121.26810466,
                  -121.53835868, -121.80772139], dtype='float64')


LATS1 = np.array([84.60636068, 86.98555849, 88.49911968, 88.90233394, 88.23555366,
                  87.41630911, 86.64939216, 85.94959841, 85.30839168, 84.71507626,
                  84.16010932, 83.63544439, 83.134311, 82.65092035, 82.18020003,
                  81.71757085, 81.25875744, 80.79962022, 80.33599603, 79.86353437,
                  79.37751496, 78.87262831, 78.3426943, 77.78028072, 77.1761612,
                  76.51850934, 75.7916446, 74.97397798, 74.03443589, 72.92573674,
                  71.57038281, 69.82683886, 67.40109717, 67.40262435, 67.40391783,
                  67.40497766, 67.40580383, 67.40639633, 67.02954233, 67.0250228,
                  67.02027475, 67.01529816, 67.01009298, 65.54008634, 65.53040248,
                  65.52050827, 65.51040361, 65.5000884, 63.11784823, 63.10408595,
                  63.09013761, 63.07600304, 63.06168202, 63.04717431, 59.96903733,
                  59.95211276, 59.93502649, 59.91777818, 59.90036748, 56.32015551,
                  56.30110949, 56.28192387, 56.2625982, 56.24313203, 52.30373269,
                  52.28317872, 52.26250411, 52.24170831, 52.22079079, 52.19975098,
                  48.00087356, 47.97913047, 47.95728212, 47.93532791, 47.91326719,
                  43.52474747, 43.50222327, 43.47960726, 43.45689875, 36.33799057,
                  37.20036236, 37.78169599, 38.21030843, 38.54535234, 38.81811012,
                  39.04703598, 39.24386487, 39.41648483, 39.57043268, 39.70973443,
                  39.83740624, 39.95576757, 40.06664499, 40.17150924, 40.27157024,
                  40.36784474, 40.46120554, 40.55241811, 40.64216823, 40.73108281,
                  40.81974518, 40.90870493, 40.99848114, 41.08955592, 41.18235086,
                  41.27717143, 41.37408581, 41.47266125, 41.57136466, 41.66608254,
                  41.74594256, 41.7785075, 49.55914071, 49.58470236, 49.61025462,
                  49.63579838, 49.66133452, 54.64219867, 54.66774027, 54.69326579,
                  54.71877613, 54.74427214, 59.70475085, 59.73023799, 59.75570099,
                  59.7811407, 59.80655795, 64.73651686, 64.7618849, 64.78721844,
                  64.8125183, 64.83778527, 64.86302015, 69.74262141, 69.76768539,
                  69.79270121, 69.81766961, 69.84259129, 74.62676182, 74.65121077,
                  74.67558881, 74.69989647, 74.72413429, 79.28634129, 79.30909484,
                  79.33173384, 79.35425837, 79.37666853, 79.39896436, 83.26233755,
                  83.27861631, 83.2946717, 83.31050269, 83.32610823, 84.6298463,
                  84.62560888, 84.62104645, 84.61615971], dtype='float64')

LONS2 = np.array([-174.41109502,  167.84584132,  148.24213696,  130.10334782,
                  115.7074828,  105.07369809,   97.28481583,   91.4618503,
                  86.98024241,   83.4283141,   80.53652225,   78.1253594,
                  76.07228855,   74.29143113,   72.72103408,   71.31559576,
                  70.04080412,   68.87020177,   67.78293355,   66.76218577,
                  65.79407472,   64.86682945,   63.97016605,   63.09478077,
                  62.23190558,   61.37287373,   60.50863405,   59.62912286,
                  58.72232744,   57.77268809,   56.75796498,   55.6419694,
                  54.36007027,   54.18827993,   40.87426478,   41.41762911,
                  41.15660793,   40.9331126,   40.73252665,   40.54677784,
                  40.37092304,   40.20150965,   40.0358693,   39.87175642,
                  39.70713409,   39.54002703,   39.36840323,   39.1900621,
                  39.00251256,   38.80282499,   38.58743647,   38.35188019,
                  38.09039231,   37.79531831,   37.45618154,   37.05815986,
                  36.57947382,   35.98665163,   35.22533847,   34.20085643,
                  32.73220377,   30.42514135,   26.23397747,   16.29417395,
                  -23.91719576, -102.71481425, -122.5294795, -129.09284487,
                  -126.27614959, -173.52330869], dtype='float64')

LATS2 = np.array([83.23214786, 84.90973645, 85.62529048, 85.74243351, 85.52147568,
                  85.13874302, 84.69067959, 84.22338069, 83.75720094, 83.30023412,
                  82.85480916, 82.42053485, 81.9957309, 81.57810129, 81.16504231,
                  80.75376801, 80.34133891, 79.92463458, 79.50028749, 79.0645828,
                  78.61332046, 78.14162813, 77.64370408, 77.11245516, 76.5389713,
                  75.91173559, 75.21538754, 74.42869094, 73.52099029, 72.44554294,
                  71.12561977, 69.42093758, 67.03973793, 67.05289493, 67.39935232,
                  67.40770791, 69.8341456, 71.57844446, 72.93459921, 74.04414258,
                  74.98457279, 75.80317362, 76.53102217, 77.1897121, 77.79492994,
                  78.3585095, 78.88968633, 79.39590402, 79.88335693, 80.35737249,
                  80.8226939, 81.28370137, 81.74459732, 82.20957417, 82.68298027,
                  83.16949849, 83.67435372, 84.20356848, 84.76429067, 85.36521771,
                  86.01711637, 86.73327122, 87.5286869, 88.40887156, 89.21959299,
                  88.71884272, 87.09172665, 84.6670132, 84.62589504, 83.31497555], dtype='float64')

LONS3 = np.array([-8.66259458, -6.20984986, 15.99813586, 25.41134052, 33.80598414,
                  48.28641356, 49.55596283, 45.21769275, 43.95449327, 30.04053601,
                  22.33028017, 13.90584249, -5.59290326, -7.75625031], dtype='float64')

LATS3 = np.array([66.94713585, 67.07854554, 66.53108388, 65.27837805, 63.50223596,
                  58.33858588, 57.71210872, 55.14964148, 55.72506407, 60.40889798,
                  61.99561474, 63.11425455, 63.67173255, 63.56939058], dtype='float64')

AREA_DEF_EURON1 = AreaDefinition('euron1', 'Northern Europe - 1km',
                                 '', {'proj': 'stere', 'ellps': 'WGS84',
                                      'lat_0': 90.0, 'lon_0': 0.0, 'lat_ts': 60.0},
                                 3072, 3072, (-1000000.0, -4500000.0, 2072000.0, -1428000.0))


def assertNumpyArraysEqual(self, other):
    if self.shape != other.shape:
        raise AssertionError("Shapes don't match")
    if not np.allclose(self, other):
        raise AssertionError("Elements don't match!")


def get_n20_orbital():
    """Return the orbital instance for a given set of TLEs for NOAA-20.
    From 16 October 2018.
    """
    tle1 = "1 43013U 17073A   18288.00000000  .00000042  00000-0  20142-4 0  2763"
    tle2 = "2 43013 098.7338 224.5862 0000752 108.7915 035.0971 14.19549169046919"
    return Orbital('NOAA-20', line1=tle1, line2=tle2)


def get_n19_orbital():
    """Return the orbital instance for a given set of TLEs for NOAA-19.
    From 16 October 2018.
    """
    tle1 = "1 33591U 09005A   18288.64852564  .00000055  00000-0  55330-4 0  9992"
    tle2 = "2 33591  99.1559 269.1434 0013899 353.0306   7.0669 14.12312703499172"
    return Orbital('NOAA-19', line1=tle1, line2=tle2)


def get_mb_orbital():
    """Return orbital for a given set of TLEs for MetOp-B.

    From 2021-02-04
    """
    tle1 = "1 38771U 12049A   21034.58230818 -.00000012  00000-0  14602-4 0 9998"
    tle2 = "2 38771  98.6992  96.5537 0002329  71.3979  35.1836 14.21496632434867"
    return Orbital("Metop-B", line1=tle1, line2=tle2)


class TestPass(unittest.TestCase):

    def setUp(self):
        """Set up"""
        self.n20orb = get_n20_orbital()
        self.n19orb = get_n19_orbital()

    def test_pass_instrument_interface(self):

        tstart = datetime(2018, 10, 16, 2, 48, 29)
        tend = datetime(2018, 10, 16, 3, 2, 38)

        instruments = set(('viirs', 'avhrr', 'modis', 'mersi', 'mersi2'))
        for instrument in instruments:
            overp = Pass('NOAA-20', tstart, tend, orb=self.n20orb, instrument=instrument)
            self.assertEqual(overp.instrument, instrument)

        instruments = set(('viirs', 'avhrr', 'modis'))
        overp = Pass('NOAA-20', tstart, tend, orb=self.n20orb, instrument=instruments)
        self.assertEqual(overp.instrument, 'avhrr')

        instruments = set(('viirs', 'modis'))
        overp = Pass('NOAA-20', tstart, tend, orb=self.n20orb, instrument=instruments)
        self.assertEqual(overp.instrument, 'viirs')

        instruments = set(('amsu-a', 'mhs'))
        self.assertRaises(TypeError, Pass, self,
                          'NOAA-20', tstart, tend, orb=self.n20orb, instrument=instruments)

    def tearDown(self):
        """Clean up"""
        pass


class TestSwathBoundary(unittest.TestCase):

    def setUp(self):
        """Set up"""
        self.n20orb = get_n20_orbital()
        self.n19orb = get_n19_orbital()
        self.mborb = get_mb_orbital()
        self.euron1 = AREA_DEF_EURON1
        self.antarctica = create_area_def(
            "antarctic",
            {'ellps': 'WGS84', 'lat_0': '-90', 'lat_ts': '-60',
             'lon_0': '0', 'no_defs': 'None', 'proj': 'stere',
             'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
            width=1000, height=1000,
            area_extent=(-4008875.4031, -4000855.294,
                         4000855.9937, 4008874.7048))
        self.arctica = create_area_def(
            "arctic",
            {'ellps': 'WGS84', 'lat_0': '90', 'lat_ts': '60',
             'lon_0': '0', 'no_defs': 'None', 'proj': 'stere',
             'type': 'crs', 'units': 'm', 'x_0': '0', 'y_0': '0'},
            width=1000, height=1000,
            area_extent=(-4008875.4031, -4000855.294,
                         4000855.9937, 4008874.7048))

    def test_swath_boundary(self):

        tstart = datetime(2018, 10, 16, 2, 48, 29)
        tend = datetime(2018, 10, 16, 3, 2, 38)

        overp = Pass('NOAA-20', tstart, tend, orb=self.n20orb, instrument='viirs')
        overp_boundary = SwathBoundary(overp)

        boundary_contour = overp_boundary.contour()

        assertNumpyArraysEqual(boundary_contour[0], LONS1)
        assertNumpyArraysEqual(boundary_contour[1], LATS1)

        tstart = datetime(2018, 10, 16, 4, 29, 4)
        tend = datetime(2018, 10, 16, 4, 30, 29, 400000)

        overp = Pass('NOAA-20', tstart, tend, orb=self.n20orb, instrument='viirs')
        overp_boundary = SwathBoundary(overp, frequency=200)

        boundary_contour = overp_boundary.contour()

        assertNumpyArraysEqual(boundary_contour[0], LONS2)
        assertNumpyArraysEqual(boundary_contour[1], LATS2)

        # NOAA-19 AVHRR:
        tstart = datetime.strptime('20181016 04:00:00', '%Y%m%d %H:%M:%S')
        tend = datetime.strptime('20181016 04:01:00', '%Y%m%d %H:%M:%S')

        overp = Pass('NOAA-19', tstart, tend, orb=self.n19orb, instrument='avhrr')
        overp_boundary = SwathBoundary(overp, frequency=500)

        boundary_contour = overp_boundary.contour()

        assertNumpyArraysEqual(boundary_contour[0], LONS3)
        assertNumpyArraysEqual(boundary_contour[1], LATS3)

        overp = Pass('NOAA-19', tstart, tend, orb=self.n19orb, instrument='avhrr/3')
        overp_boundary = SwathBoundary(overp, frequency=500)

        boundary_contour = overp_boundary.contour()

        assertNumpyArraysEqual(boundary_contour[0], LONS3)
        assertNumpyArraysEqual(boundary_contour[1], LATS3)

        overp = Pass('NOAA-19', tstart, tend, orb=self.n19orb, instrument='avhrr-3')
        overp_boundary = SwathBoundary(overp, frequency=500)

        boundary_contour = overp_boundary.contour()

        assertNumpyArraysEqual(boundary_contour[0], LONS3)
        assertNumpyArraysEqual(boundary_contour[1], LATS3)

    def test_swath_coverage(self):

        # NOAA-19 AVHRR:
        tstart = datetime.strptime('20181016 03:54:13', '%Y%m%d %H:%M:%S')
        tend = datetime.strptime('20181016 03:55:13', '%Y%m%d %H:%M:%S')

        overp = Pass('NOAA-19', tstart, tend, orb=self.n19orb, instrument='avhrr')

        cov = overp.area_coverage(self.euron1)
        self.assertEqual(cov, 0)

        overp = Pass('NOAA-19', tstart, tend, orb=self.n19orb, instrument='avhrr', frequency=80)

        cov = overp.area_coverage(self.euron1)
        self.assertEqual(cov, 0)

        tstart = datetime.strptime('20181016 04:00:00', '%Y%m%d %H:%M:%S')
        tend = datetime.strptime('20181016 04:01:00', '%Y%m%d %H:%M:%S')

        overp = Pass('NOAA-19', tstart, tend, orb=self.n19orb, instrument='avhrr')

        cov = overp.area_coverage(self.euron1)
        self.assertAlmostEqual(cov, 0.103526, 5)

        overp = Pass('NOAA-19', tstart, tend, orb=self.n19orb, instrument='avhrr', frequency=100)

        cov = overp.area_coverage(self.euron1)
        self.assertAlmostEqual(cov, 0.103526, 5)

        overp = Pass('NOAA-19', tstart, tend, orb=self.n19orb, instrument='avhrr/3', frequency=133)

        cov = overp.area_coverage(self.euron1)
        self.assertAlmostEqual(cov, 0.103526, 5)

        overp = Pass('NOAA-19', tstart, tend, orb=self.n19orb, instrument='avhrr', frequency=300)

        cov = overp.area_coverage(self.euron1)
        self.assertAlmostEqual(cov, 0.103526, 5)

        # ASCAT and AVHRR on Metop-B:
        tstart = datetime.strptime("2019-01-02T10:19:39", "%Y-%m-%dT%H:%M:%S")
        tend = tstart + timedelta(seconds=180)
        tle1 = '1 38771U 12049A   19002.35527803  .00000000  00000+0  21253-4 0 00017'
        tle2 = '2 38771  98.7284  63.8171 0002025  96.0390 346.4075 14.21477776326431'

        mypass = Pass('Metop-B', tstart, tend, instrument='ascat', tle1=tle1, tle2=tle2)
        cov = mypass.area_coverage(self.euron1)
        self.assertAlmostEqual(cov, 0.322936, 5)

        mypass = Pass('Metop-B', tstart, tend, instrument='avhrr', tle1=tle1, tle2=tle2)
        cov = mypass.area_coverage(self.euron1)
        self.assertAlmostEqual(cov, 0.357324, 5)

        tstart = datetime.strptime("2019-01-05T01:01:45", "%Y-%m-%dT%H:%M:%S")
        tend = tstart + timedelta(seconds=60*15.5)

        tle1 = '1 43010U 17072A   18363.54078832 -.00000045  00000-0 -79715-6 0  9999'
        tle2 = '2 43010  98.6971 300.6571 0001567 143.5989 216.5282 14.19710974 58158'

        mypass = Pass('FENGYUN 3D', tstart, tend, instrument='mersi2', tle1=tle1, tle2=tle2)
        cov = mypass.area_coverage(self.euron1)

        self.assertAlmostEqual(cov, 0.786836, 5)

    def test_arctic_is_not_antarctic(self):

        tstart = datetime(2021, 2, 3, 16, 28, 3)
        tend = datetime(2021, 2, 3, 16, 31, 3)

        overp = Pass('Metop-B', tstart, tend, orb=self.mborb, instrument='avhrr')

        cov_south = overp.area_coverage(self.antarctica)
        cov_north = overp.area_coverage(self.arctica)

        assert cov_north == 0
        assert cov_south != 0

    def tearDown(self):
        """Clean up"""
        pass


class TestPassList(unittest.TestCase):

    def setUp(self):
        """Set up"""
        pass

    def test_meos_pass_list(self):
        orig = ("  1 20190105 FENGYUN 3D  5907 52.943  01:01:45 n/a   01:17:15 15:30  18.6 107.4 -- "
                "Undefined(Scheduling not done 1546650105 ) a3d0df0cd289244e2f39f613f229a5cc D")

        tstart = datetime.strptime("2019-01-05T01:01:45", "%Y-%m-%dT%H:%M:%S")
        tend = tstart + timedelta(seconds=60 * 15.5)

        tle1 = '1 43010U 17072A   18363.54078832 -.00000045  00000-0 -79715-6 0  9999'
        tle2 = '2 43010  98.6971 300.6571 0001567 143.5989 216.5282 14.19710974 58158'

        mypass = Pass('FENGYUN 3D', tstart, tend, instrument='mersi2', tle1=tle1, tle2=tle2)

        coords = (10.72, 59.942, 0.1)
        meos_format_str = mypass.print_meos(coords, line_no=1)

        self.assertEqual(meos_format_str, orig)

    def test_generate_metno_xml(self):
        import xml.etree.ElementTree as ET
        root = ET.Element("acquisition-schedule")

        orig = ('<acquisition-schedule><pass satellite="FENGYUN 3D" aos="20190105010145" los="20190105011715" '
                'orbit="5907" max-elevation="52.943" asimuth-at-max-elevation="107.385" asimuth-at-aos="18.555" '
                'pass-direction="D" satellite-lon-at-aos="76.204" satellite-lat-at-aos="80.739" '
                'tle-epoch="20181229125844.110848" /></acquisition-schedule>')

        tstart = datetime.strptime("2019-01-05T01:01:45", "%Y-%m-%dT%H:%M:%S")
        tend = tstart + timedelta(seconds=60 * 15.5)

        tle1 = '1 43010U 17072A   18363.54078832 -.00000045  00000-0 -79715-6 0  9999'
        tle2 = '2 43010  98.6971 300.6571 0001567 143.5989 216.5282 14.19710974 58158'

        mypass = Pass('FENGYUN 3D', tstart, tend, instrument='mersi2', tle1=tle1, tle2=tle2)

        coords = (10.72, 59.942, 0.1)
        mypass.generate_metno_xml(coords, root)

        self.assertEqual(ET.tostring(root).decode("utf-8"), orig)

    def tearDown(self):
        """Clean up"""
        pass


def suite():
    """The suite for test_satpass
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSwathBoundary))
    mysuite.addTest(loader.loadTestsFromTestCase(TestPass))
    mysuite.addTest(loader.loadTestsFromTestCase(TestPassList))

    return mysuite
