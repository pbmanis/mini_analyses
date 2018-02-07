"""
Analysis

"""
from __future__ import print_function
import os
import read_protocol as rp
import clements_bekkers as cb
import numpy as np
import matplotlib.pyplot as mpl
import scipy.signal

# basedatadir = '/Volumes/Pegasus/ManisLab_Data3/Sullivan_Chelsea/miniIPSCs'
# 
# datasets = {#0: {'dir': '2017.04.25_000/slice_000/cell_002/minis_007', 'thr': 3.5, 'rt': 0.35, 'decay': 6.},
# #            1: {'dir': '2017.05.02_000/slice_000/cell_000/', 'prots': [0,1,2]'thr': 2.0, 'rt': 0.35, 'decay': 5.},
#             2: {'dir': '2017.05.02_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 2.5, 'rt': 0.35, 'decay': 5.},
# 
#             }f
from  CS_CHL1_minis import *
basedatadir = basepath

dt = 0.1
for d in datasets.keys():
    dprot = 2
#    print( ('minis_03d' % datasets[d]['prots'][dprot]))
    fn = os.path.join(basedatadir, datasets[d]['dir'], ('minis_{0:03d}'.format(datasets[d]['prots'][dprot])))
    data, time_base, dt = rp.readPhysProtocol(fn, records=None)
    data = data.asarray()*1e12
    aj = cb.AndradeJonas()
    dt = dt * 1000. # convert to msec
    time_base = time_base*1000.
    aj.make_template(0.35, datasets[d]['decay'], np.max(time_base), dt, sign=-1.)
    intv = []
    ampd = []
    mwin = int(2*(4.)/dt)
    order = int(4/dt)
    for i in range(data.shape[0]):
        data[i] = data[i] - data[i].mean()
        aj.deconvolve(data[i], np.max(time_base), dt=dt, thresh=datasets[d]['thr'], llambda=10., order=7)
        aj.plots(np.max(time_base), dt, data[i])
        # summarize data
        intervals = np.diff(aj.timebase[aj.onsets])
        intv.extend(intervals)
        pks = []
        amps = []
        for j in range(len(aj.onsets)):
             p =  scipy.signal.argrelextrema(aj.sign*data[i][aj.onsets[j]:(aj.onsets[j]+mwin)], np.greater, order=order)[0]
             if len(p) > 0:
                 pks.extend([int(p[0]+aj.onsets[j])])
                 amp = aj.sign*data[i][pks[-1]] - aj.sign*data[i][aj.onsets[j]]
                 amps.extend([amp])

        # mpl.figure()
        # mpl.plot(aj.timebase, data[i], 'k-')
        # mpl.plot(aj.timebase[pks], data[i][pks], 'ro')
        # mpl.show()
        # exit()
        ampd.extend(amps)

#    print (len(ampd))
    f, ax = mpl.subplots(2, 1)
    ax[0].hist(ampd, 50)
    ax[1].hist(intv, 50)
    print('N events: {0:7d}'.format(len(intv)))
    print('Intervals: {0:7.1f} ms SD = {1:.1f} Frequency: {2:7.1f} Hz'.format(np.mean(intv), np.std(intv), 1e3/np.mean(intv)))
    print('Amplitude: {0:7.1f} pA SD = {1:.1f}'.format(np.mean(ampd), np.std(ampd)))
    mpl.show()