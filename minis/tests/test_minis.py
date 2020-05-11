# -*- encoding: utf-8 -*-
"""
Test fixture for minis_methods
Provide synthesis of data, and run each of the tests. 

"""

import sys
import numpy as np
import scipy.signal
import dataclasses
from dataclasses import dataclass, field
import typing
from typing import Union, Dict, List

from minis.util import UserTester
# from cnmodel.protocols import SynapseTest

import minis.minis_methods as MM

# import pyqtgraph as pg



def def_taus():
    return [0.001, 0.010]  # in seconds


def def_template_taus():
    return [0.002, 0.005]


@dataclass
class EventParameters:
    dt: float = 2e-5  # msec
    tdur:float = 1e-1
    maxt: float = 10.0  # sec
    meanrate: float = 10.0  # Hz
    amp: float = 20e-12  # Amperes
    ampvar: float = 5e-12  # Amperes (variance)
    noise: float = 4e-12# 0.0  # Amperes, gaussian nosie
    threshold: float = 2.5  # threshold used in the analyses
    LPF:Union[None, float] = None # low-pass filtering applied to data, f in Hz or None
    HPF:Union[None, float] = None # high pass filter (sometimes useful)
    taus: list = field(
        default_factory=def_taus
    )  # rise, fall tau in msec for test events
    template_taus: list = field(
        default_factory=def_template_taus
    )  # initial tau values in search
    sign: int = -1  # sign: 1 for positive, -1 for negative
    mindur: float = 1e-3  # minimum length of an event in seconds
    bigevent: Union[
        None, dict
    ] = None  # whether to include a big event; if so, the event is {'t', 'I'} np arrays.
    expseed: Union[int, None] = 1  # starting seed for intervals
    noiseseed: Union[int, None] = 1  # starting seed for background noise


def printPars(pars):
    print(dir(pars))
    d = dataclasses.asdict(pars)
    for k in d.keys():
        print("k: ", k, " = ", d[k])


# these are the tests that will be run

def test_ZeroCrossing():
    MinisTester(method='ZC')
    
    
def test_ClementsBekkers():
    MinisTester(method='CB')


def test_AndradeJonas():
    MinisTester(method='AJ')
    


def generate_testdata(
    pars: dataclass,
    baseclass: Union[object, None] = None,
    func: Union[object, None] = None,
):
    """
        meanrate is in Hz(events/second)
        maxt is in seconds
        bigevent is a dict {'t': delayinsec, 'I': amplitudeinA}
    """
    if baseclass is None and func is not None:
        raise ValueError("Need base class definition")

    timebase = np.arange(0.0, pars.maxt, pars.dt)  # in ms
    t_psc = np.arange(0.0, pars.tdur, pars.dt)  # time base for single event template in ms
    if func is None:  # make double-exp event
        tau_1 = pars.taus[0]  # ms
        tau_2 = pars.taus[1]  # ms
        Apeak = pars.amp  # pA
        Aprime = (tau_2 / tau_1) ** (tau_1 / (tau_1 - tau_2))
        g = Aprime * (-np.exp(-t_psc / tau_1) + np.exp((-t_psc / tau_2)))
        gmax = np.max(g)
        # print(np.min(g))
        # printPars(pars)
        # print("gmax: ", gmax)
        g = pars.sign * g * pars.amp / gmax
        # print(f'max g: {np.min(g):.6e}')
    else:  # use template from the class
        baseclass._make_template()
        gmax = np.min(baseclass.template)
        g = sign * amp * baseclass.template / gmax
        # print('gmaxb: ', np.max(gmax))

    testpsc = np.zeros(timebase.shape)
    if pars.expseed is None:
        eventintervals = np.random.exponential(
            1.0 / pars.meanrate, int(pars.maxt * pars.meanrate)
        )
    else:
        np.random.seed(pars.expseed)
        eventintervals = np.random.exponential(
            1.0 / pars.meanrate, int(pars.maxt * pars.meanrate)
        )
    eventintervals = eventintervals[eventintervals < 10.0]
    events = np.cumsum(eventintervals)
    if pars.bigevent is not None:
        events = np.append(events, pars.bigevent["t"])
        events = np.sort(events)
    t_events = events[events < pars.maxt]  # time of events with exp distribution
    i_events = np.array([int(x / pars.dt) for x in t_events])
    testpsc[i_events] = np.random.normal(1.0, pars.ampvar / pars.amp, len(i_events))
    if pars.bigevent is not None:
        ipos = int(pars.bigevent["t"] / pars.dt)  # position in array
        testpsc[ipos] = pars.bigevent["I"]
    testpsc = scipy.signal.convolve(testpsc, g, mode="full")[: timebase.shape[0]]

    if pars.noise > 0:
        if pars.noiseseed is None:
            testpscn = testpsc + np.random.normal(0.0, pars.noise, testpsc.shape)
        else:
            np.random.seed(pars.noiseseed)
            testpscn = testpsc + np.random.normal(0.0, pars.noise, testpsc.shape)
    else:
        testpscn = testpsc
    return timebase, testpsc, testpscn, i_events, t_events




def run_ZeroCrossing(pars=None, bigevent:bool=False, plot: bool = False) -> object:
    """
    Do some tests of the CB protocol and plot
    """
    if pars is None:
        pars = EventParameters()
    minlen = int(pars.mindur / pars.dt)
    if bigevent:
        pars.bigevent = {"t": 1.0, "I": 20.0}
    for i in range(1):
        timebase, testpsc, testpscn, i_events, t_events = generate_testdata(pars)
        zc = MM.ZCFinder()
        zc.setup(
            dt=pars.dt, tau1=pars.template_taus[0], tau2=pars.template_taus[1], sign=pars.sign
        )  # dt=dt, tau1=0.001, tau2=0.005, sign=-1)
        events = zc.find_events(
            testpscn, data_nostim=None, thresh=pars.threshold, minLength=minlen,
            lpf=pars.LPF, hpf=pars.HPF,
        )
        print('# events in template: ', len(t_events))
        
    if plot:
        zc.plots(title="Zero Crossings")
    return zc


def run_ClementsBekkers(pars:dataclass=None, bigevent:bool=False, plot:bool=False) -> object:
    """
    Do some tests of the CB protocol and plot
    """
    if pars is None:
        pars = EventParameters()
    if bigevent:
        pars.bigevent = {"t": 1.0, "I": 20.0}
    for i in range(1):
        pars.noiseseed = i * 47
        pars.expseed = i
        timebase, testpsc, testpscn, i_events, t_events = generate_testdata(pars)
        cb = MM.ClementsBekkers()
        pars.baseclass = cb
        cb.setup(
            tau1=pars.template_taus[0],
            tau2=pars.template_taus[1],
            dt=pars.dt,
            delay=0.0,
            template_tmax= 5*pars.template_taus[1],
            sign=pars.sign,
        )
        cb._make_template()
        cb.cbTemplateMatch(testpscn, threshold=pars.threshold, lpf=pars.LPF)
        print('# events in template: ', len(t_events))
    if plot:
        cb.plots(title="Clements Bekkers")
    return cb


def run_AndradeJonas(pars:dataclass=None, bigevent:bool=False, plot: bool = False) -> object:
    # sign = -1
    # trace_dur = 10.  # seconds
    # dt = 5e-5
    # amp = 100e-12
    if pars is None:
        pars = EventParameters()
    if bigevent:
        pars.bigevent = {"t": 1.0, "I": 20.0}
    for i in range(1):
        aj = MM.AndradeJonas()
        aj.setup(
            tau1=pars.template_taus[0],
            tau2=pars.template_taus[1],
            dt=pars.dt,
            delay=0.0,
            template_tmax = pars.maxt,  # taus are for template
            sign=pars.sign,
            risepower=4.0,
        )
        # generate test data
        pars.baseclass = aj
        pars.noiseseed = i * 47
        pars.expseed = i
        timebase, testpsc, testpscn, i_events, t_events = generate_testdata(pars)

        aj.deconvolve(
            testpscn - np.mean(testpscn),
            thresh=pars.threshold,
            lpf=pars.LPF,
            llambda=1,
            order=int(0.001 / pars.dt),
        )
        print('# events in template: ', len(t_events))
        
    if plot:
        aj.summarize(aj.data)
        aj.plots(events=None, title="AJ")  # i_events)
    return aj

class MiniTestMethods():
    def __init__(self, method:str='cb', plot:bool=False):
        self.plot = plot
        self.testmethod = method

    def run_test(self):

        pars = EventParameters()
        pars.LPF=1500

        if self.testmethod in ["ZC", "zc"]:
            pars.threshold=0.9
            pars.mindur=1e-3
            pars.HPF=20.
            result = run_ZeroCrossing(pars, plot=True)
            print('# detected events: ', len(result.allevents))
            if self.plot:
                zct = np.arange(0, result.allevents.shape[1]*result.dt, result.dt)
                for a in range(len(result.allevents)):
                    mpl.plot(zct, result.allevents[a])
                mpl.show()
        if self.testmethod in ["CB", "cb"]:
            result = run_ClementsBekkers(pars, plot=True)
            print(len(result.allevents))
            if self.plot:
                for a in range(len(result.allevents)):
                    mpl.plot(result.t_template, result.allevents[a])
                    mpl.show()
        if self.testmethod in ["AJ", "aj"]:
            result = run_AndradeJonas(pars, plot=True)
            print(len(result.allevents))
            ajt = result.t_template[0:result.allevents.shape[1]]
            if self.plot:
                for a in range(len(result.allevents)):
                    mpl.plot(ajt, result.allevents[a])
                    mpl.show()            
        # if self.testmethod in ["all", "ALL"]:
        #     run_ZeroCrossing(pars, plot=True)
        #     run_ClementsBekkers(pars, plot=True)
        #     run_AndradeJoans(pars, plot=True)

        # print(dir(result))
        # print('result figure:: ', result.P.figure_handle)
        # print('result figure:: ', dir(result.P.figure_handle.figure))
        # print('result figure:: ', result.P)
        testresult = {'onsets': result.onsets,
                      'peaks': result.peaks,
                      'amplitudes': result.amplitudes,
                      'fitresult': result.fitresult,
                      'fitted_tau1': result.fitted_tau1,
                      'fitted_tau2': result.fitted_tau2,
                      'risepower': result.risepower,
                      'risetenninety': result.risetenninety,
                      'decaythirtyseven': result.decaythirtyseven,  
                  }
        return testresult 

class MinisTester(UserTester):
    def __init__(self, method):
        self.TM = None
        self.figure = None
        UserTester.__init__(self, "%s" % method, method)

            
    def run_test(self, method):

        info = {}
        
        self.TM = MiniTestMethods(method=method)
        test_result = self.TM.run_test()

        if 'figure' in list(test_result.keys()):
            self.figure = test_result['figure']
        # # seed random generator using the name of this test
        # seed = "%s_%s" % (pre, post)
        #
        # pre_cell = make_cell(pre)
        # post_cell = make_cell(post)
        #
        # n_term = convergence.get(pre, {}).get(post, None)
        # if n_term is None:
        #     n_term = 1
        # st = SynapseTest()
        # st.run(pre_cell.soma, post_cell.soma, n_term, seed=seed)
        # if self.audit:
        #     st.show_result()
        #
        # info = dict(
        #     rel_events=st.release_events(),
        #     rel_timings=st.release_timings(),
        #     open_prob=st.open_probability(),
        #     event_analysis=st.analyze_events(),
        #     )
        # self.st = st
        #
        # #import weakref
        # #global last_syn
        # #last_syn = weakref.ref(st.synapses[0].terminal.relsi)
        #
        return test_result
    
    def assert_test_info(self, *args, **kwds):
        try:
            super(MinisTester, self).assert_test_info( *args, **kwds)
        finally:
            if self.figure is not None:
                del(self.figure)

    

if __name__ == "__main__":
    if len(sys.argv[0]) > 1:
        testmethod = sys.argv[1]
    if testmethod not in ["ZC", "CB", "AJ", "zc", "cb", "aj", "all", "ALL"]:
        print("Test for %s method is not implemented" % testmethod)
        exit(1)
    else:
        # set up for plotting
        import matplotlib

        rcParams = matplotlib.rcParams
        rcParams["svg.fonttype"] = "none"  # No text as paths. Assume font installed.
        rcParams["pdf.fonttype"] = 42
        rcParams["ps.fonttype"] = 42
        rcParams["text.usetex"] = False
        import matplotlib.pyplot as mpl
        import matplotlib.collections as collections
        import warnings  # need to turn off a scipy future warning.

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message="UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans",
        )
        import pylibrary.plotting.plothelpers as PH

        pars = EventParameters()
        pars.LPF=1500
        if testmethod in ["ZC", "zc"]:
            pars.threshold=0.9
            pars.mindur=1e-3
            pars.HPF=20.
            zc = run_ZeroCrossing(pars, plot=True)
            print('# detected events: ', len(zc.allevents))
            zct = np.arange(0, zc.allevents.shape[1]*zc.dt, zc.dt)
            for a in range(len(zc.allevents)):
                mpl.plot(zct, zc.allevents[a])
            mpl.show()
        if testmethod in ["CB", "cb"]:
            cb = run_ClementsBekkers(pars, plot=True)
            print(len(cb.allevents))
            for a in range(len(cb.allevents)):
                mpl.plot(cb.t_template, cb.allevents[a])
            mpl.show()
        if testmethod in ["AJ", "aj"]:
            aj = run_AndradeJonas(pars, plot=True)
            print(len(aj.allevents))
            ajt = aj.t_template[0:aj.allevents.shape[1]]
            for a in range(len(aj.allevents)):
                mpl.plot(ajt, aj.allevents[a])
            mpl.show()            
        if testmethod in ["all", "ALL"]:
            run_ZeroCrossing(pars, plot=True)
            run_ClementsBekkers(pars, plot=True)
            run_AndradeJoans(pars, plot=True)

    #    pg.show()
    # if sys.flags.interactive == 0:
    #     pg.QtGui.QApplication.exec_()
