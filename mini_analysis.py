from __future__ import print_function

"""
Analysis of miniature synaptic potentials
Provides measure of event amplitude distribution and event frequency distribution
Driven by imported dictionary
Requires a bunch of Manis' support routines, including ephysanalysis module to
read acq4 files; pylibrary utilities, digital filtering. 


"""
import sys
import os
#import read_protocol as rp
import clements_bekkers as cb
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.gridspec as gridspec
import scipy.signal
import scipy.stats
import pylibrary.Utility as PU
import digital_filters as DF
import ephysanalysis as EP
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
rc('text', usetex=False)
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

# basedatadir = '/Volumes/Pegasus/ManisLab_Data3/Sullivan_Chelsea/miniIPSCs'
# 
# datasets = {#0: {'dir': '2017.04.25_000/slice_000/cell_002/minis_007', 'thr': 3.5, 'rt': 0.35, 'decay': 6.},
# #            1: {'dir': '2017.05.02_000/slice_000/cell_000/', 'prots': [0,1,2]'thr': 2.0, 'rt': 0.35, 'decay': 5.},
#             2: {'dir': '2017.05.02_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 2.5, 'rt': 0.35, 'decay': 5.},
# 
#             }f
from  CS_CHL1_minis import *
basedatadir = basepath


def analyze_all(fofilename=None):
    with PdfPages(fofilename) as pdf:
        for id, mouse in enumerate(datasets.keys()):
            analyze_one(datasets[mouse], pdf, maxprot=10)


def analyze_one(mousedata, pdf, maxprot=0, testrun=False):
    dt = 0.1

    print (mousedata)
    excl = mousedata['exclist']  # get the exclusion list
    cell_summary={'intervals': [], 'amplitudes': []}

    # for all of the protocols that are included for this cell (identified by mouse and a letter)
    for nprot, dprot in enumerate(mousedata['prots']):
        if nprot > maxprot:
            return
        exclude_traces = []
        if len(mousedata['exclist']) > 0:  # any traces to exclude?
            if dprot in mousedata['exclist'].keys():
                exclude_traces = mousedata['exclist'][dprot]
                
        print('exclude traces: ',  exclude_traces)
        #continue
        # try:
        #     print ('Dataset {0:s}   dprot: {1:d}, prot: {2:d}'.format(d, dprot, datasets[d]['prots'][dprot]))
        # except:
        #     print("******failed with: d, dprot, dataset[d]['prots']", d, dprot, datasets[d]['prots'])
        if testrun:
            continue
        fn = os.path.join(basedatadir, mousedata['dir'], ('minis_{0:03d}'.format(mousedata['prots'][dprot])))
        print ('fn: ', fn)
        dainfo = fn
        a = EP.acq4read.Acq4Read(dataname='Clamp1.ma')
        a.setProtocol(fn)
        try:
            a.getData()
        except:
            print('Get data failed... ')
            continue
        #a.plotClampData(all=True)
        #continue
        #
        #data, time_base, dt = rp.readPhysProtocol(fn, records=None)
        data = np.array(a.data_array)
        time_base = a.time_base
        dt = a.sample_interval  # in sec... so convert (later)
        print('dt: ', dt)
        data = data*1e12
        aj = cb.AndradeJonas()
        dt = dt * 1000. # convert to msec
        time_base = time_base*1000.
        aj.make_template(0.35, mousedata['decay'], np.max(time_base), dt, sign=-1.)

        mwin = int(2*(4.)/dt)
        order = int(4/dt)
        ntraces = data.shape[0]
        tracelist = range(ntraces)
        # if exclude_traces is not None:
       #      for et in exclude_traces:
       #          tracelist.remove(et)

        fig = mpl.figure(1)  # one subplot for each trace
        fig.set_size_inches(8.5, 11.0, forward=True)
        fig.tight_layout()
        hht = 3
        ax0 = mpl.subplot2grid((ntraces+hht,1), (0,0), colspan=2, rowspan=ntraces, fig=fig)
        axIntvls = mpl.subplot2grid((ntraces+hht, 2), (ntraces, 0), colspan=1, rowspan=hht, fig=fig)
        axAmps = mpl.subplot2grid((ntraces+hht, 2), (ntraces, 1), colspan=1, rowspan=hht, fig=fig)
        fig.suptitle(dainfo)

        yspan = 40.
        print ('tracelist: ', tracelist, ' exclude: ', exclude_traces)
        ax0.set_xlim(-500., np.max(a.time_base)*1000.)
        nused = 0
        # for all of the traces collected in this protocol run
        for i in tracelist:
            yp = (ntraces-i)*yspan
            data[i] = data[i] - data[i].mean()
            odata = data[i].copy()
            if i in exclude_traces:
                ax0.plot(a.time_base*1000., odata + yp ,'y-', linewidth=0.25, alpha=0.25)
                ax0.text(-400., yp, '%d' % i)
                continue
            nused = nused + 1
            
            frs, freqs = PU.pSpectrum(data[i], samplefreq=1000./dt)
            # NotchFilter(signal, notchf=[60.], Q=90., QScale=True, samplefreq=None):
            data[i] = DF.NotchFilter(data[i], [60., 120., 180., 240., 300., 360, 420., 
                            480., 660., 780., 1020., 1140., 1380., 1500., 4000.], Q=20., samplefreq=1000./dt)
            dfilt = DF.SignalFilter_HPFButter(np.pad(data[i], (len(data[i]), len(data[i])), 'mean'), 2.5, 1000./dt, NPole=4)
            dfilt = DF.SignalFilter_LPFButter(dfilt, 2800., 1000./dt, NPole=4)
            #data[i] = PU.SignalFilter(data[i], 3500., 12., 1000./dt)
            frsfilt, freqsf = PU.pSpectrum(dfilt, samplefreq=1000./dt)
            data[i] = dfilt[len(data[i]):2*len(data[i])]
            aj.deconvolve(data[i], np.max(time_base), dt=dt, thresh=mousedata['thr']*1.2, llambda=10., order=7)
            #aj.plots(np.max(time_base), dt, data[i])
            # summarize data
            intervals = np.diff(aj.timebase[aj.onsets])
            cell_summary['intervals'].extend(intervals)
            pks = []
            amps = []
            for j in range(len(aj.onsets)):
                 p =  scipy.signal.argrelextrema(aj.sign*data[i][aj.onsets[j]:(aj.onsets[j]+mwin)], np.greater, order=order)[0]
                 if len(p) > 0:
                     pks.extend([int(p[0]+aj.onsets[j])])
                     amp = aj.sign*data[i][pks[-1]] - aj.sign*data[i][aj.onsets[j]]
                     amps.extend([amp])
            cell_summary['amplitudes'].extend(amps)
 
            ax0.plot(aj.timebase, odata + yp ,'c-', linewidth=0.25, alpha=0.25)
            ax0.text(-400, yp, '%d' % i)
            ax0.plot(aj.timebase, data[i] + yp, 'k-', linewidth=0.25)
            ax0.plot(aj.timebase[pks], data[i][pks] + yp, 'ro', markersize=3)


        if nused == 0:
            continue
        # summarize the event distribution and the amplitude distribution

        # amplitudes:
        histBins = 50
        nsamp = 1
        nevents = len(cell_summary['amplitudes'])
        amp, ampbins, amppa= axAmps.hist(cell_summary['amplitudes'], histBins, alpha=0.5, normed=True)

        # normal distribution
        ampnorm = scipy.stats.norm.fit(cell_summary['amplitudes'])  #
        print('Amplitude Normfit: mean {0:.3f}   sigma: {1:.3f}'.format(ampnorm[0], ampnorm[1]))
        # ampDis = np.zeros((nsamp, nevents))  # 20 trials
        # for i in range(nsamp):
        #     ampDis[i,:] = scipy.stats.norm.rvs(loc=ampnorm[0], scale=ampnorm[1], size=nevents)
        # ampDis = np.mean(ampDis, axis=0)
        # print ('ampdis rvs: mean: %f  std:  %f' % (np.mean(ampDis), np.std(ampDis)))
        # ta, tb, tap = axAmps.hist(ampDis, bins=ampbins, histtype='step', color='r')
        x = np.linspace(scipy.stats.norm.ppf(0.01, loc=ampnorm[0], scale=ampnorm[1]),
                         scipy.stats.norm.ppf(0.99, loc=ampnorm[0], scale=ampnorm[1]), 100)
        axAmps.plot(x, scipy.stats.norm.pdf(x, loc=ampnorm[0], scale=ampnorm[1]),
                'r-', lw=3, alpha=0.6, label='norm pdf')

       # axAmps.plot([np.mean(cell_summary['amplitudes']), np.mean(cell_summary['amplitudes'])], [0., np.max(ta)], 'g-')

        # skewed normal distriubution
        ampskew = scipy.stats.skewnorm.fit(cell_summary['amplitudes'])
        print('ampskew: mean: {0:.4f} skew:{1:4f}  scale/sigma: {2:4f} '.format(ampskew[1], ampskew[0], 2*ampskew[2]))
        x = np.linspace(scipy.stats.skewnorm.ppf(0.002, a=ampskew[0], loc=ampskew[1], scale=ampskew[2]),
                         scipy.stats.skewnorm.ppf(0.995, a=ampskew[0], loc=ampskew[1], scale=ampskew[2]), 100)
        axAmps.plot(x, scipy.stats.skewnorm.pdf(x, a=ampskew[0], loc=ampskew[1], scale=ampskew[2]),
                'm-', lw=3, alpha=0.6, label='skewnorm pdf')

        # gamma distriubution
        ampgamma = scipy.stats.gamma.fit(cell_summary['amplitudes'])
        print('ampgamma: mean: {0:.4f} gamma:{1:4f}  scale/sigma: {2:4f} '.format(ampgamma[1], ampgamma[0], 2*ampgamma[2]))
        x = np.linspace(scipy.stats.gamma.ppf(0.002, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]),
                         scipy.stats.gamma.ppf(0.995, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]), 100)
        axAmps.plot(x, scipy.stats.gamma.pdf(x, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]),
                'g-', lw=3, alpha=0.6, label='gammapdf')
        axAmps.legend(fontsize=8)
        
        an, bins, patches = axIntvls.hist(cell_summary['intervals'], histBins)
        nintvls = len(cell_summary['intervals'])
        expDis = scipy.stats.expon.rvs(scale=np.std(cell_summary['intervals']), loc=0, size=nintvls)
        axIntvls.hist(expDis, bins=bins, histtype='step', color='r')
        print('N events: {0:7d}'.format(nintvls))
        print('Intervals: {0:7.1f} ms SD = {1:.1f} Frequency: {2:7.1f} Hz'.format(np.mean(cell_summary['intervals']),
                np.std(cell_summary['intervals']), 1e3/np.mean(cell_summary['intervals'])))
        print('Amplitude: {0:7.1f} pA SD = {1:.1f}'.format(np.mean(cell_summary['amplitudes']), np.std(cell_summary['amplitudes'])))
        # test if interval distribtuion is poisson:

        stats = scipy.stats.kstest(expDis,'expon', args=((np.std(cell_summary['intervals'])),), alternative = 'two-sided')
        print('KS test for intervals Exponential: statistic: {0:.5f}  p={1:.3e}'.format(stats.statistic, stats.pvalue))
        stats_amp = scipy.stats.kstest(expDis,'norm', 
                args=(np.mean(cell_summary['amplitudes']),
                      np.std(cell_summary['amplitudes'])), alternative = 'two-sided')
        print('KS test for Normal amplitude: statistic: {0:.5f}  p={1:.3e}'.format(stats_amp.statistic, stats_amp.pvalue))
        #dir(stats))

        if pdf is None:
            mpl.show()
        else:
            pdf.savefig()
            mpl.close()
        
if __name__ == '__main__':
    nargs = len(sys.argv)
    if nargs == 1:
        analyze_all(fofilename = 'all.pdf')
    else:
        analyze_one(datasets[sys.argv[1]], pdf=None, maxprot=10)