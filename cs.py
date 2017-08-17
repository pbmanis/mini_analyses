"""
Analysis

"""
from __future__ import print_function
import sys
import os
import read_protocol as rp
import minis_methods as minis
import numpy as np
import matplotlib.pyplot as mpl
import scipy.signal

basedatadir = '/Volumes/Pegasus/ManisLab_Data3/Sullivan_Chelsea/miniIPSCs'

datasets = {'m1a': {'dir': '2017.04.25_000/slice_000/cell_002', 'prots': [7], 'thr': 3.5, 'rt': 0.35, 'decay': 6., 'G': 'F/+'},
            'm2a': {'dir': '2017.05.02_000/slice_000/cell_000/', 'prots': [0,1,2], 'thr': 2.0, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm2b': {'dir': '2017.05.02_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 2.0, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm2c': {'dir': '2017.05.02_000/slice_000/cell_002', 'prots': [0,1,2], 'thr': 2.0, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm3a': {'dir': '2017.05.04_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 2.0, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm4a': {'dir': '2017.05.05_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm5a': {'dir': '2017.05.11_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm5b': {'dir': '2017.05.11_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm6a': {'dir': '2017.07.05_000/slice_000/cell_001', 'prots': [2,3,4], 'thr': 2.5, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm7a': {'dir': '2017.07.06_000/slice_000/cell_000', 'prots': [1,2,3], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm7b': {'dir': '2017.07.06_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/F'},
            'm7c': {'dir': '2017.07.06_000/slice_000/cell_002', 'prots': [1,2], 'thr': 2.0, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm7d': {'dir': '2017.07.06_000/slice_000/cell_003', 'prots': [0,1,6], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
           # compared to others, e is an unstable recording
           # 'm7e': {'dir': '2017.07.06_000/slice_000/cell_004', 'prots': [0,1,2], 'thr': 2.5, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm8a': {'dir': '2017.07.07_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm8b': {'dir': '2017.07.07_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            # M9: some data has big noise, not acceptable
            #'m9a': {'dir': '2017.07.19_000/slice_000/cell_000', 'prots': [2,3,4], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            # m9b: protocols 0 and 2 have noise, not acceptable; 1 is ok
            'm9b': {'dir': '2017.07.19_000/slice_000/cell_001', 'prots': [1], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm9c': {'dir': '2017.07.19_000/slice_000/cell_002', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            # incomple data for m9d:
            # 'm9d': {'dir': '2017.07.19_000/slice_000/cell_003', 'prots': [0], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            # m10a: runs 1 & 2 have unacceptable noise
            'm10a': {'dir': '2017.07.27_000/slice_000/cell_000', 'prots': [0], 'thr': 2.0, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm10b': {'dir': '2017.07.27_000/slice_000/cell_001', 'prots': [0], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm10c': {'dir': '2017.07.27_000/slice_000/cell_002', 'prots': [0], 'thr': 2.25, 'rt': 0.35, 'decay': 3.5, 'G': 'F/F'},
            # m10c, run 2: suspicious bursts 
            'm10d': {'dir': '2017.07.27_000/slice_000/cell_003', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},
            'm10e': {'dir': '2017.07.27_000/slice_000/cell_004', 'prots': [1], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},

            }



def do_one_protocol(ds, dprot, plots=True):
    fn = os.path.join(basedatadir, datasets[ds]['dir'], ('minis_{0:03d}'.format(datasets[ds]['prots'][dprot])))
    try:
        data, time_base, dt = rp.readPhysProtocol(fn, records=None)
    except:
        print("Incomplete protocol: {:s}".format(fn))
        raise ValueError('bad data')
    data = data.asarray()*1e12
    aj = minis.AndradeJonas()
    dt = dt * 1000. # convert to msec
    time_base = time_base*1000.
    maxt = np.max(time_base)
    aj.setup(tau1=datasets[ds]['rt'], tau2=datasets[ds]['decay'], tmax=maxt, dt=dt, sign=-1.)
    intv = []
    ampd = []
    mwin = int(2*(4.)/dt)
    order = int(4/dt)
    for i in range(data.shape[0]):
        data[i] = data[i] - data[i].mean()
        aj.deconvolve(data[i], thresh=datasets[ds]['thr'], llambda=20., order=7)
        ampd.extend(aj.amplitudes)
        intv.extend(aj.intervals)
        if i == 0 and plots:
            aj.plots()
#    print len(ampd)
    # if plots:
        # f, ax = mpl.subplots(2, 1)
        # try:
        #     ax[0].hist(ampd, 50)
        # except:
        #     print('ampd: ', ampd)
        # ax[1].hist(intv, 50)
    print('Dataset: {:s}'.format(fn))
    print('    N events: {0:7d}'.format(len(intv)))
    print('    Intervals: {0:7.1f} ms SD = {1:.1f} Frequency: {2:7.1f} Hz'.format(np.nanmean(intv), np.nanstd(intv), 1e3/np.mean(intv)))
    print('    Amplitude: {0:7.1f} pA SD = {1:.1f}'.format(np.nanmean(ampd), np.nanstd(ampd)))
    return np.nanmean(intv), np.nanmean(ampd)
 
def runall():
    gt_intd =  }
    gt_ampd = {}
    gt_mouse = []
    for d in datasets.keys():  # by cell (not mouse)
        cintv = 0.
        campd = 0.
        n = 0
        g = datasets[d]['G']
        cintv = 0.
        for dprot in range(len(datasets[d]['prots'])):
            intv, ampd = do_one_protocol(d, dprot, plots=False)
            cintv += 1000./intv
            campd += ampd
            n += 1
        if g not in gt_intd.keys():
            gt_intd[g] = [cintv/n]
        else:
            gt_intd[g].extend([cintv/n])
        if g not in gt_ampd.keys():
            gt_ampd[g] = [campd/n]
        else:
            gt_ampd[g].extend([campd/n])
        if d not in gt_mouse:
            gt_mouse.append([d])

    print gt_mouse
    for k in gt_intd.keys():
        print('Genotype: {0:s}'.format(k))
        print('intvs: ', gt_intd[k])
    for k in gt_intd.keys():
        print('Genotype: {0:s}'.format(k))
        print('amps: ', gt_ampd[k])
        
    #mpl.show()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        runall()
    else:
        ds = sys.argv[1]
        for dprot in range(len(datasets[ds]['prots'])):
            do_one_protocol(ds, dprot)
        mpl.show()

          