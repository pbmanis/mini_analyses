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
from scipy.optimize import curve_fit
from collections import OrderedDict

import commands
computer_name = commands.getoutput('scutil --get ComputerName')
print('computer name: {0}'.format(computer_name))
if computer_name in ['Tamalpais']:
    basedir = ('/Volumes/PBM_004/data/NCAM-miniIPSCs')
elif computer_name in ['Lytle']:
    basedir = ('/Volumes/Pegasus/ManisLab_Data3/Sullivan_Chelsea/miniIPSCs')
else:
    raise ValueError('Computer name not in known list of names to set base path')


"""
Summary 14 mice
m#(1) means mouse #N, (#) is the number of cells
7 F/+  (m1(1), m2(3), m3(1), m4(1), m5(2), m9(2), m13(4)) = 14
7 F/F (m6(1), m7(4), m8(2), m10(5), m11(1), M12(1), m14(1)) = 15
"""
datasets = {'m1a': {'dir': '2017.04.25_000/slice_000/cell_001', 'prots': [0,1,3], 'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/+'},
            'm1b': {'dir': '2017.04.25_000/slice_000/cell_002', 'prots': [7], 'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/+'},
            'm2a': {'dir': '2017.05.02_000/slice_000/cell_000/', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.32, 'decay': 5., 'G': 'F/+'},
            'm2b': {'dir': '2017.05.02_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm2c': {'dir': '2017.05.02_000/slice_000/cell_002', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm3a': {'dir': '2017.05.04_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm4a': {'dir': '2017.05.05_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm5a': {'dir': '2017.05.11_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm5b': {'dir': '2017.05.11_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm6a': {'dir': '2017.07.05_000/slice_000/cell_001', 'prots': [2,3,4], 'thr': 2.5, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},  # unusually low rate
            'm7a': {'dir': '2017.07.06_000/slice_000/cell_000', 'prots': [1,2,3], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm7b': {'dir': '2017.07.06_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/F'},
            'm7c': {'dir': '2017.07.06_000/slice_000/cell_002', 'prots': [1,2], 'thr': 2.0, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm7d': {'dir': '2017.07.06_000/slice_000/cell_003', 'prots': [0,1,6], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
           # compared to others, e is an unstable recording
           # 'm7e': {'dir': '2017.07.06_000/slice_000/cell_004', 'prots': [0,1,2], 'thr': 2.5, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm8a': {'dir': '2017.07.07_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm8b': {'dir': '2017.07.07_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            # M9: some data has big noise, not acceptable
            #'m9a': {'dir': '2017.07.19_000/slice_000/cell_000', 'prots': [2,3,4], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            # m9b: protocols 0 and 2 have noise, not acceptable; 1 is ok
            'm9b': {'dir': '2017.07.19_000/slice_000/cell_001', 'prots': [1], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm9c': {'dir': '2017.07.19_000/slice_000/cell_002', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            # incomple data for m9d11:
            # 'm9d': {'dir': '2017.07.19_000/slice_000/cell_003', 'prots': [0], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            # m10a: runs 1 & 2 have unacceptable noise
            'm10a': {'dir': '2017.07.27_000/slice_000/cell_000', 'prots': [0], 'thr': 2.0, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm10b': {'dir': '2017.07.27_000/slice_000/cell_001', 'prots': [0], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm10c': {'dir': '2017.07.27_000/slice_000/cell_002', 'prots': [0], 'thr': 2.25, 'rt': 0.35, 'decay': 3.5, 'G': 'F/F'},
            # m10c, run 2: suspicious bursts
            'm10d': {'dir': '2017.07.27_000/slice_000/cell_003', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},
            #'m10e': {'dir': '2017.07.27_000/slice_000/cell_004', 'prots': [1], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},  # unstable and bursty
#
#  more:
#
            'm11a': {'dir': '2017.08.10_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.25, 'rt': 0.35, 'decay': 6, 'G': 'F/F'},
            'm12a': {'dir': '2017.08.11_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.0, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},
            'm13b': {'dir': '2017.08.15_000/slice_000/cell_001', 'prots': [1,2], 'thr': 1.0, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm13c': {'dir': '2017.08.15_000/slice_000/cell_002', 'prots': [1,2], 'thr': 3.0, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},  # protocol minis_000 not very good - removed
            #'m13d': {'dir': '2017.08.15_000/slice_000/cell_003', 'prots': [0,1,2], 'thr': 2.0, 'rt': 0.35, 'decay': 4., 'G': 'F/+'}, # quite variable rate in runs 0 and 1 - dropped entire recording
            'm13e': {'dir': '2017.08.15_000/slice_000/cell_004', 'prots': [3], 'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},  # runs 1 and 2 had burstiness and instability - dropped
            # m13f has weird bursts - exclude
            #'m13f': {'dir': '2017.08.15_000/slice_000/cell_005', 'prots': [0,1,2], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            # cells 006 and 007 are no good for 8/15
            # cells 000, 001 002 ng for 8/16
            'm14d': {'dir': '2017.08.16_000/slice_000/cell_003', 'prots': [1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},  # dropped run 0 - had unstable traces (baseline drift)
            
            }

class Summary():
    def __init__(self):
        pass
        
    def doubleexp(self, x, A, tau_1, tau_2, dc):
        tm = A * ((1-(np.exp(-x/tau_1)))**4 * np.exp((-x/tau_2))) + dc
        return tm
    
    def fit_average_event(self, x, y):

        init_vals = [-20., 0.5, 5., 0.]
        best_vals, covar = curve_fit(self.doubleexp, x, y, p0=init_vals)
#        print ('best vals: ', best_vals)
        self.fitresult = best_vals
        self.best_fit = self.doubleexp(self.tbx, best_vals[0], best_vals[1], best_vals[2], best_vals[3])
        # lmfit version - fails for odd reason
        # dexpmodel = Model(self.doubleexp)
        # params = dexpmodel.make_params(A=-10., tau_1=0.5, tau_2=4.0, dc=0.)
        # self.fitresult = dexpmodel.fit(self.avgevent[tsel:], params, x=self.avgeventtb[tsel:])
        # print(self.fitresult.fit_report())
        self.best_vals = best_vals
        self.tau1 = best_vals[1]
        self.tau2 = best_vals[2]

    def do_one_protocol(self, ds, dprot, plots=False):
        fn = os.path.join(basedir, datasets[ds]['dir'], ('minis_{0:03d}'.format(datasets[ds]['prots'][dprot])))
        try:
            data, time_base, dt = rp.readPhysProtocol(fn, records=None)
        except:
            print("Incomplete protocol: {:s}".format(fn))
            raise ValueError('bad data')
        print('ds: {0:s}   dprot: {1:d}'.format(ds, dprot))
        data = data.asarray()*1e12
        aj = minis.AndradeJonas()
        dt = dt * 1000. # convert to msec
        time_base = time_base*1000.
        maxt = np.max(time_base)
        aj.setup(tau1=datasets[ds]['rt'], tau2=datasets[ds]['decay'], tmax=maxt, dt=dt, sign=-1.)
        intv = []
        ampd = []
        events = []
        mwin = int(2*(4.)/dt)
        order = int(4/dt)
        
        for i in range(data.shape[0]):  # typically 10
            data[i] = data[i] - data[i].mean()
            aj.deconvolve(data[i], thresh=datasets[ds]['thr'], llambda=20., order=7)
            ampd.extend(aj.amplitudes)
            intv.extend(aj.intervals)
            if i == 0: # get event shape
                print(aj.allevents.shape)
            events.extend(aj.allevents)
            if i >= 0 and plots:
                aj.plots()
    #    print len(ampd)
        if plots:
            f, ax = mpl.subplots(2, 1)
            try:
                ax[0].hist(ampd, 50)
            except:
                pass
                # print('ampd: ', ampd)
            ax[1].hist(intv, 50)

        print("")
        print('Dataset: {:s}'.format(fn))
        print('    N events: {0:7d}'.format(len(intv)))
        print('    Intervals: {0:7.1f} ms SD = {1:.1f} Frequency: {2:7.1f} Hz'.format(np.nanmean(intv), np.nanstd(intv), 1e3/np.mean(intv)))
        print('    Amplitude: {0:7.1f} pA SD = {1:.1f}'.format(np.nanmean(ampd), np.nanstd(ampd)))
        # print('         tau1: {0:7.1f} pA SD = {1:.1f}'.format(np.nanmean(tau1), np.nanstd(tau1)))
       #  print('         tau2: {0:7.1f} pA SD = {1:.1f}'.format(np.nanmean(tau2), np.nanstd(tau2)))
       #   
        #return np.nanmean(intv), np.nanmean(ampd), np.nanmean(tau1), np.nanmean(tau2)
        self.delay = aj.delay
        self.intv = intv
        self.ampd = ampd
        self.events = np.array(events)
        self.aj = aj
        self.dt = dt
        #print('events: ', self.events.shape)
        
 
    def runall(self):
        gt_rate =  OrderedDict()
        gt_ampd = OrderedDict()
        gt_tau1 = OrderedDict()
        gt_tau2 = OrderedDict()
        gt_mouse = []
        for d in datasets.keys():  # by cell (not mouse)
            cintv = []
            campd = []
            ctau1 = 0.
            ctau2 = 0.
            n = 0
            g = datasets[d]['G']
#            cintv = 0.
            for i, dprot in enumerate(range(len(datasets[d]['prots']))):
                self.do_one_protocol(d, dprot, plots=False)
                
                cintv.extend(self.intv)
                campd.extend(self.ampd)
                n += 1
                if i == 0:
                    allev = self.events
                else:
                    allev = np.concatenate((allev, self.events))
#                print('allev shape: ', allev.shape)
#            print(cintv)
            # now get average event shape, and fit it
            avg = np.mean(allev, axis=0)
            t = np.arange(0, self.dt*avg.shape[0], self.dt)
            tsel = np.argwhere(t >= 6.0)[0][0]
            self.tbx = t[:-tsel]
            self.fit_average_event(t[:-tsel], avg[tsel:])            
            
            if g not in gt_rate.keys():
                gt_rate[g] = [1000./np.nanmean(cintv)]
            else:
                gt_rate[g].extend([1000./np.nanmean(cintv)])
            
            if g not in gt_ampd.keys():
                gt_ampd[g] = [np.nanmean(campd)]
            else:
                gt_ampd[g].extend([np.nanmean(campd)])
            
            if g not in gt_tau1.keys():
                gt_tau1[g] = [self.tau1]
            else:
                gt_tau1[g].extend([self.tau1])
            
            if g not in gt_tau2.keys():
                gt_tau2[g] = [self.tau2]
            else:
                gt_tau2[g].extend([self.tau2])
        

            if d not in gt_mouse:
                gt_mouse.append([d])

        print( gt_mouse)
        for k in gt_rate.keys():
            print('Genotype: {0:s}'.format(k))
            print('Rate: ', gt_rate[k])
        for k in gt_rate.keys():  # keeps same order.
            print('Genotype: {0:s}'.format(k))
            print('amps: ', gt_ampd[k])
        for k in gt_rate.keys():
            print('Genotype: {0:s}'.format(k))
            print('tau1: ', gt_tau1[k])
        for k in gt_rate.keys():
            print('Genotype: {0:s}'.format(k))
            print('tau2: ', gt_tau2[k])
        
    #mpl.show()


if __name__ == '__main__':
    s = Summary()
    if len(sys.argv) == 1:
        s.runall()
    else:
        ds = sys.argv[1]
        for i, dprot in enumerate(range(len(datasets[ds]['prots']))):
            s.do_one_protocol(ds, dprot, plots=True)
            if i == 0:
                allev = s.events
            else:
                allev = np.concatenate((allev, s.events))
        avg = np.mean(allev, axis=0)
        t = np.arange(0, s.dt*avg.shape[0], s.dt)
        tsel = np.argwhere(t >= 5.0 + s.delay)[0][0]
        s.tbx = t[:-tsel]
        s.fit_average_event(t[:-tsel], avg[tsel:])
        print('    Tau1: %7.3f  Tau2: %7.3f' % (s.best_vals[1], s.best_vals[2]))
#        s.best_fit = [-30., 0.5, 4., -3]
        mpl.plot(t,  avg, 'k')
        mpl.plot(t[tsel:], s.best_fit, 'r--')
#        mpl.plot(t[tsel:], s.doubleexp(t[:-tsel], s.best_fit[0] , s.best_fit[1], s.best_fit[2], s.best_fit[3]), 'r--')
        mpl.show()

          