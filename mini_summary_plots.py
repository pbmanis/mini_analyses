from __future__ import print_function

"""
mini_summary_plots.py
Simple plots of data points for mini analysis from mini_analysis.py

Paul B. Manis, 3/2018
"""

import os
import sys
import pickle
import numpy as np
from collections import OrderedDict
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Helvetica']
rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as mpl
import seaborn as sns

import pylibrary.PlotHelpers as PH

fn = 'summarydata.p'  # data file to plot from - should hold a dict of mouse entries
# each mouse m entry in d has the following keys:
# ['amplitudes', 'genotype', 'intervals', 'mouse', 'protocols', 'eventcounts']

fh = open(fn, 'rb')
d = pickle.load(fh)
fh.close()

print ('Mice/cells: ', d.keys())

gtypes = []
amps = {}
meanamps = {}
intvls = {}

for i, m in enumerate(d.keys()):
    gt = d[m]['genotype']
    if gt not in gtypes:
        gtypes.append(gt)
        amps[gt] = []
        meanamps[gt] = []
        intvls[gt] = []
    if i == 0:
        print( d[m].keys())
    amps[gt].append(d[m]['amplitude_midpoint'])
    meanamps[gt].append(np.mean(d[m]['amplitudes']))
    intvls[gt].append(np.mean(d[m]['intervals']))

# print (amps)
# print (meanamps)
# print (intvls)

sizer = OrderedDict([ ('A', {'pos': [0.12, 0.35, 0.15, 0.7]}),
                      ('B', {'pos': [0.55, 0.35, 0.15, 0.7]}),
                     ])  # dict elements are [left, width, bottom, height] for the axes in the plot.
n_panels = len(sizer.keys())
gr = [(a, a+1, 0, 1) for a in range(0, n_panels)]   # just generate subplots - shape does not matter
axmap = OrderedDict(zip(sizer.keys(), gr))
P = PH.Plotter((n_panels, 1), axmap=axmap, label=True, figsize=(5., 3.))
P.resize(sizer)  # perform positioning magic

P.axdict['A'].plot(np.ones(len(intvls['WT'])), 1000./np.array(intvls['WT']), 'ko', markersize=4.0)
P.axdict['A'].plot(1+np.ones(len(intvls['CHL1'])), 1000./np.array(intvls['CHL1']), 'bs', markersize=4.0)
P.axdict['A'].set_xlim(0.5, 2.5)
P.axdict['A'].set_ylim(0., 20.)
P.axdict['A'].set_xlabel('Genotype')
P.axdict['A'].set_ylabel('Mean Event Frequency (Hz)')
P.axdict['B'].plot(np.ones(len(amps['WT'])), amps['WT'], 'ko', markerfacecolor='k', markeredgecolor='k', markersize=4.0, markeredgewidth=1)
P.axdict['B'].plot(1.0 + np.ones(len(amps['CHL1'])), amps['CHL1'], 'bs', markerfacecolor='b', markeredgecolor='b', markersize=4.0, markeredgewidth=1)
P.axdict['B'].plot(0.2 + np.ones(len(meanamps['WT'])), meanamps['WT'], 'ko', markerfacecolor='w', markeredgecolor='k', markersize=4.0, markeredgewidth=1)
P.axdict['B'].plot(1.2 + np.ones(len(meanamps['CHL1'])), meanamps['CHL1'], 'bs', markerfacecolor='w', markeredgecolor='b', markersize=4.0, markeredgewidth=1)
P.axdict['B'].set_xlabel('Genotype')
P.axdict['B'].set_ylabel('Amplitude (pA)')
P.axdict['B'].set_xlim(0.5, 2.5)

P.axdict['B'].set_xlim(0.5, 2.5)
P.axdict['B'].set_ylim(0., 30.)
PH.formatTicks(P.axdict['A'], font='Helvetica')
PH.formatTicks(P.axdict['B'], font='Helvetica')

mpl.savefig('msummary.pdf')
mpl.show() 

