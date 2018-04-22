from __future__ import print_function

"""
Classes for methods that do analysis of miniature synaptic potentials

Current implementations are ClementsBekkers and AndradeJonas

Test run timing:
cb: 0.175 s (with cython version of algorithm); misses overlapping events
aj: 0.028 s, plus gets overlapping events

July 2017

"""
import numpy as np
import scipy.signal
import matplotlib.pyplot as mpl

import timeit
from scipy.optimize import curve_fit
from numba import jit
import digital_filters as dfilt
import clembek
import pylibrary.PlotHelpers as PH


class MiniAnalyses():
    def __init__(self):
        """
        Base class for Clements-Bekkers and Andrade-Jonas methods
        Provides template generation, and summary analyses
        Allows use of common methods between different algorithms
        """
        self.risepower = 4.
        pass

    def _make_template(self):
        """
        Private function: make template when it is needed
        """
        tau_1,  tau_2 = self.taus
        t_psc = np.arange(0,  self.template_tmax,  self.dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        self.template = np.zeros_like(t_psc)
        tm = 1./Aprime * ((1-(np.exp(-t_psc/tau_1)))**self.risepower * np.exp((-t_psc/tau_2)))


        
        if self.idelay > 0:
            self.template[self.idelay:] = tm[:-self.idelay]  # shift the template
        else:
            self.template = tm
        if self.sign > 0:
            self.template_amax = np.max(self.template)
        else:
            self.template = -self.template
            self.template_amax = np.min(self.template)    

    def summarize(self,  data, order=11):
        """
        compute intervals,  peaks and ampitudes for all found events
        """
        self.intervals = np.diff(self.timebase[self.onsets])  # event intervals
        i_decay_pts = int(self.taus[1]/self.dt)  # decay window time (points)
        self.peaks = []
        self.smpkindex = []
        self.smoothed_peaks = []
        self.amplitudes = []
        self.averaged = False  # set flags in case of no events found
        self.fitted = False
        ndata = len(data)
        avgwin = 5 # int(1.0/self.dt)  # 5 point moving average window for peak detection
        mwin = int(2*(4.)/self.dt)
        order = int(4./self.dt)
   #     print('onsets: ', self.onsets)
        if self.sign > 0:
            nparg = np.greater
        else:
            nparg = np.less
        if len(self.onsets) > 0:  # original events
            acceptlist = []
            for j in range(len(data[self.onsets])):
                if self.sign > 0 and self.eventstartthr is not None:
                    if self.data[self.onsets[j]] < self.eventstartthr:
                        continue
                if self.sign < 0 and self.eventstartthr is not None:
                    if self.data[self.onsets[j]] > -self.eventstartthr:
                        continue
                p =  scipy.signal.argrelextrema(data[self.onsets[j]:(self.onsets[j]+mwin)], nparg, order=order)[0]
                if len(p) > 0:
                    self.peaks.extend([int(p[0]+self.onsets[j])])
                    amp = self.sign*(self.data[self.peaks[-1]] - data[self.onsets[j]])

                    self.amplitudes.extend([amp])
                    i_end = i_decay_pts + self.onsets[j] # distance from peak to end
                    i_end = min(ndata, i_end)  # keep within the array limits
                    if j < len(self.onsets)-1:
                        if i_end > self.onsets[j+1]:
                            i_end = self.onsets[j+1]-1 # only go to next event start
                    move_avg, n = moving_average(data[self.onsets[j]:i_end], n=min(avgwin, len(data[self.onsets[j]:i_end])))
                    if self.sign > 0:
                        pk = np.argmax(move_avg) # find peak of smoothed data
                    else:
                        pk = np.argmin(move_avg)
                    self.smoothed_peaks.extend([move_avg[pk]])  # smoothed peak
                    self.smpkindex.extend([self.onsets[j]+pk])
                    acceptlist.append(j)
            if len(acceptlist) < len(self.onsets):
                print('Trimmed %d events' % (len(self.onsets)-len(acceptlist)))
                self.onsets = self.onsets[acceptlist] # trim to only the accepted values
                
            self.average_events()
            self.fit_average_event()
        else:
            print('No events found')
            return

    def average_events(self):
        # compute average event with length of template
        tdur = np.max((np.max(self.taus)*3.0, 5.0))  # go 3 taus or 5 ms past event
        tpre = 0 # self.taus[0]*10.
        self.tpre = tpre
        self.avgnpts = int((tpre+tdur)/self.dt)  # points for the average
        npre = int(tpre/self.dt) # points for the pre time
        npost = int(tdur/self.dt)
        avg = np.zeros(self.avgnpts)
        self.allevents = np.zeros((len(self.onsets),  self.avgnpts))
        k = 0
        pkt = 0 # np.argmax(self.template)
        for j, i in enumerate(self.onsets):
            ix = i + pkt # self.idelay
            if (ix + npost) < len(self.data) and (ix - npre) >= 0:
                self.allevents[k,:] = self.data[ix-npre:ix+npost]
                k = k + 1
        if k > 0:
            self.allevents = self.allevents[0:k, :]  # trim unused
            self.avgevent = self.allevents.mean(axis=0)
            self.avgeventtb = np.arange(self.avgevent.shape[0])*self.dt
            self.averaged = True
        else:
            self.averaged = False

    def doubleexp(self, x, t, y, risepower):
        """
        Calculate a double expoential EPSC-like waveform with the rise to a power
        to make it sigmoidal
        """
        tm = x[0] * ((1.0 - np.exp(-t/x[1]))**risepower )* np.exp((-t/x[2]))
        return tm-y
    
    def fit_average_event(self):
        """
        Fit the averaged event to a double exponential epsc-like function
        """
        if not self.averaged:  # avoid fit if averaging has not been done
            return
        #tsel = np.argwhere(self.avgeventtb > self.tpre)[0]  # only fit data in event,  not baseline
        tsel = 0  # use whole averaged trace
        self.tsel = tsel
        
        # setup for fit
        init_vals = [self.sign*10.,  0.5,  4.]
        bounds  = [(-4000., 0.075, 0.2), (4000., 10., 50.)]
        res = scipy.optimize.least_squares(self.doubleexp, init_vals,
                        bounds=bounds, args=(self.avgeventtb[self.tsel:], self.avgevent[self.tsel:], self.risepower))
        self.fitresult = res.x
        self.best_fit = self.doubleexp(self.fitresult, self.avgeventtb[self.tsel:],
            np.zeros_like(self.avgeventtb[self.tsel:]), risepower=self.risepower)
        # lmfit version - fails for odd reason
        # dexpmodel = Model(self.doubleexp)
        # params = dexpmodel.make_params(A=-10.,  tau_1=0.5,  tau_2=4.0,  dc=0.)
        # self.fitresult = dexpmodel.fit(self.avgevent[tsel:],  params,  x=self.avgeventtb[tsel:])
        self.Amplitude = self.fitresult[0]
        self.tau1 = self.fitresult[1]
        self.tau2 = self.fitresult[2]
        self.DC = 0. # best_vals[3]
        
        ave = self.sign*self.avgevent
        ipk = np.argmax(ave)
        pk = ave[ipk]
        p10 = 0.1*pk
        p90 = 0.9*pk
        p37 = 0.37*pk
        i10 = np.argmin(np.fabs(ave[:ipk]-p10))
        i90 = np.argmin(np.fabs(ave[:ipk]-p90))
        i37 = np.argmin(np.fabs(ave[ipk:]-p37))
        self.risetenninety = self.dt*(i90-i10)
        self.decaythirtyseven = self.dt*(i37-ipk)
        
        self.fitted = True

    def plots(self,  events=None,  title=None):
        """
        Plot the results from the analysis and the fitting
        """
        data = self.data
        P = PH.regular_grid(3 , 1, order='columns', figsize=(8., 6), showgrid=False,
                        verticalspacing=0.08, horizontalspacing=0.08,
                        margins={'leftmargin': 0.07, 'rightmargin': 0.20, 'topmargin': 0.03, 'bottommargin': 0.1},
                        labelposition=(-0.12, 0.95))
        ax = P.axarr
        ax = ax.ravel()
        PH.nice_plot(ax)
        for i in range(1,2):
            ax[i].get_shared_x_axes().join(ax[i],  ax[0])
        # raw traces, marked with onsets and peaks
        tb = self.timebase[:len(data)]
        ax[0].plot(tb,  data,  'k-',  linewidth=0.75, label='Data')  # original data
        ax[0].plot(tb[self.onsets],  data[self.onsets],  'k^',  
                        markersize=6,  markerfacecolor=(1,  1,  0,  0.8),  label='Onsets')
        if len(self.onsets) is not None:
#            ax[0].plot(tb[events],  data[events],  'go',  markersize=5, label='Events')
#        ax[0].plot(tb[self.peaks],  self.data[self.peaks],  'r^', label=)
            ax[0].plot(tb[self.smpkindex],  self.smoothed_peaks,  'r^', label='Smoothed Peaks')
        ax[0].set_ylabel('I (pA)')
        ax[0].set_xlabel('T (ms)')
        ax[0].legend(fontsize=8, loc=2, bbox_to_anchor=(1.0, 1.0))
        
        # deconvolution trace, peaks marked (using onsets), plus threshold)
        ax[1].plot(tb[:self.Crit.shape[0]],  self.Crit, label='Deconvolution') 
        ax[1].plot([tb[0],tb[-1]],  [self.sdthr,  self.sdthr],  'r--',  linewidth=0.75, 
                label='Threshold ({0:4.2f}) SD'.format(self.sdthr))
        ax[1].plot(tb[self.onsets]-self.idelay,  self.Crit[self.onsets],  'y^', label='Deconv. Peaks')
        if events is not None:  # original events
            ax[1].plot(tb[:self.Crit.shape[0]][events],  self.Crit[events],
                    'ro',  markersize=5.)
        ax[1].set_ylabel('Deconvolution')
        ax[1].set_xlabel('T (ms)')
        ax[1].legend(fontsize=8, loc=2, bbox_to_anchor=(1.0, 1.0))
#        print (self.dt, self.template_tmax, len(self.template))
        # averaged events, convolution template, and fit
        if self.averaged:
            ax[2].plot(self.avgeventtb[:len(self.avgevent)],  self.avgevent, 'k', label='Average Event')
            maxa = np.max(self.sign*self.avgevent)
            #tpkmax = np.argmax(self.sign*self.template)
            temp_tb = np.arange(0, len(self.template)*self.dt, self.dt)
            #print(len(self.avgeventtb[:len(self.template)]), len(self.template))
            ax[2].plot(self.avgeventtb[:len(self.avgevent)],  self.sign*self.template[:len(self.avgevent)]*maxa/self.template_amax,  
                'r-', label='Template')
            ax[2].plot(self.avgeventtb[:len(self.best_fit)],  self.best_fit,  'c--', linewidth=2.0, 
                label='Best Fit (Rise Power={0:.2f}\nTau1={1:.1f} Tau2={2:.1f})'.format(self.risepower, self.tau1, self.tau2))
            if title is not None:
                P.figure_handle.suptitle(title)
            ax[2].set_ylabel('Averaged I (pA)')
            ax[2].set_xlabel('T (ms)')
            ax[2].legend(fontsize=8, loc=2, bbox_to_anchor=(1.0, 1.0))
        print('measures: ', self.risetenninety, self.decaythirtyseven)
        mpl.show()
        

@jit(nopython=True,  cache=True)
def nb_clementsbekkers(data,  template):
    ## Prepare a bunch of arrays we'll need later
    n_template = len(template)
    n_data = data.shape[0]
    n_dt = n_data - n_template
    sum_template = template.sum()
    sum_template_2 = (template*template).sum()

    data_2 = data*data
    sum_data = np.sum(data[:n_template])
    sum_data_2 = data_2[:n_template].sum()
    scale = np.zeros(n_dt)
    offset = np.zeros(n_dt)
    crit = np.zeros(n_dt)
    for i in range(n_dt):
        if i > 0:
            sum_data = sum_data + data[i+n_template] - data[i-1]
            sum_data_2 = sum_data_2 + data_2[i+n_template] - data_2[i-1]
        sum_data_template_prod = np.multiply(data[i:i+n_template],  template).sum()
        scale[i] = (
                    (sum_data_template_prod - sum_data * sum_template/n_template)/
                    (sum_template_2 - sum_template*sum_template/n_template)
                     )
        offset[i] = (sum_data - scale[i]*sum_template)/n_template
        fitted_template = template * scale[i] + offset[i]
        sse = ((data[i:i+n_template] - fitted_template)**2).sum()
        crit[i] = scale[i]/np.sqrt(sse/(n_template-1))
    DC = scale/ crit
    return(DC,  scale,  crit)


class ClementsBekkers(MiniAnalyses):
    def __init__(self):
        self.dt = None
        self.data = None
        self.template = None
        self.engine = 'cython'

    def setup(self, tau1=None,  tau2=None,  template_tmax=None,  dt=None,  
            delay=0.0,  sign=1, eventstartthr=None, risepower=4.0):
        """
        Just store the parameters - will compute when needed
        """
        assert sign in [-1, 1]
        self.sign = sign
        self.taus = [tau1,  tau2]
        self.dt = dt
        self.template_tmax = template_tmax
        self.idelay = int(delay/dt)  # points delay in template with zeros
        self.template = None  # reset the template if needed.
        self.eventstartthr = eventstartthr
        self.risepower = risepower
    
    def set_cb_engine(self, engine):
        """
        Define which detection engine to use
        cython requires compilation outised
        Numba does a JIT compilation (see routine above)
        """
        if engine == 'numba':
            self.engine = engine
        elif engine == 'cython':
            self.engine = engine
        else:
            raise ValueError('CB detection engine must be either numba or cython')

    def clements_bekkers_numba(self, data):
        self.timebase = np.arange(0.,  self.data.shape[0]*self.dt,  self.dt)
        D = data.view(np.ndarray)
        DC, self.Scale, self.Crit = nb_clementsbekkers(D,  self.template)
        
    def clements_bekkers_cython(self,  data):
        if self.template is None:
            self._make_template()    
        self.timebase = np.arange(0.,  self.data.shape[0]*self.dt,  self.dt)
        D = data.view(np.ndarray)
        T = self.template.view(np.ndarray)
        crit = np.zeros_like(D)
        scale = np.zeros_like(D)
        offset = np.zeros_like(D)
        pkl = np.zeros(10000)
        evl = np.zeros(10000)
        nout = 0
        nt = T.shape[0]
        nd = D.shape[0]
        clembek.clembek(D,  T,  self.threshold,  crit,  scale,  offset,  pkl,  evl,  nout,  self.sign,  nt,  nd)
        self.Scale = scale
        self.Crit = crit
        self.DC = offset

    def clements_bekkers(self,  data):
        """
        Implements Clements-bekkers algorithm: slides template across data,          returns array of points indicating goodness of fit.
        Biophysical Journal,  73: 220-229,  1997.
        
        Parameters
        ----------
        data : np.array (no default)
            1D data array
        
        """
        starttime = timeit.default_timer()
        if self.template is None:
            self._make_template()    

        ## Strip out meta-data for faster computation
        D = self.sign*data.view(np.ndarray)
        T = self.template.view(np.ndarray)
        self.timebase = np.arange(0.,  self.data.shape[0]*self.dt,  self.dt)
        if self.engine == 'numba':
            DC,  S,  crit = nb_clementsbekkers(D,  T)
            self.DC = DC
            self.Scale = S
            self.Crit = crit
        if self.engine == 'cython':
            self.clements_bekkers_cython(D)
        endtime = timeit.default_timer() - starttime
        print('CB {0:s} runtime: {1:.4f} s'.format(self.engine, endtime))

        self.Crit = self.sign*self.Crit  # assure that crit is positive
    
    def cbTemplateMatch(self,  data, threshold=3.0, llambda=5.0,  order=7):
        self.data = data
        self.threshold = threshold

        self.clements_bekkers(self.data)  # flip data sign if necessary
        self.Crit = self.Crit
        sd = np.std(self.Crit)
        self.sdthr = sd * self.threshold  # set the threshold
        self.above = np.clip(self.Crit,  self.sdthr,  None)
        self.onsets = scipy.signal.argrelextrema(self.above,  np.greater,  order=int(order))[0] - 1 + self.idelay
        self.summarize(self.data)
        mask = self.Crit > threshold
        diff = mask[1:] - mask[:-1]
        times = np.argwhere(diff==1)[:,  0]  ## every time we start OR stop an event


class AndradeJonas(MiniAnalyses):
    """
    Deconvolution method of Andrade/Jonas,  Biophysical Journal 2012
    Create an instance of the class (aj = AndradeJonas())
    call setup to instantiate the template and data detection sign (1 for positive, -1 for negative)
    call deconvolve to perform the deconvolution
    additional routines provide averaging and some event analysis and plotting
    
    """
    def __init__(self):
        self.template = None
        self.onsets = None
        self.timebase = None
        self.dt = None
        self.sign = 1
        self.taus = None
        self.template_max = None
        self.idelay = 0

    def setup(self,  tau1=None,  tau2=None,  template_tmax=None,  dt=None,  
            delay=0.0,  sign=1, eventstartthr=None, risepower=4.):
        """
        Just store the parameters - will compute when needed
        """
        assert sign in [-1, 1]
        self.sign = sign
        self.taus = [tau1,  tau2]
        self.dt = dt
        self.template_tmax = template_tmax
        self.idelay = int(delay/dt)  # points delay in template with zeros
        self.template = None  # reset the template if needed.
        self.eventstartthr = eventstartthr
        self.risepower = risepower

            
    def _make_template(self):
        """
        Private function: make template when it is needed
        """
        tau_1,  tau_2 = self.taus
        t_psc = np.arange(0,  self.template_tmax,  self.dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        self.template = np.zeros_like(t_psc)
        tm = 1./Aprime * ((1-(np.exp(-t_psc/tau_1)))**4 * np.exp((-t_psc/tau_2)))
        
        if self.idelay > 0:
            self.template[self.idelay:] = tm[:-self.idelay]  # shift the template
        else:
            self.template = tm
        if self.sign > 0:
            self.template_amax = np.max(self.template)
        else:
            self.template = -self.template
            self.template_amax = np.min(self.template)
            
        
    def deconvolve(self,  data,  thresh=1.0,  llambda=5.0,  order=7):
        if self.template is None:
            self._make_template()
        self.data = dfilt.SignalFilter_LPFButter(data,  3000.,  1000/self.dt,  NPole=8)
        self.timebase = np.arange(0.,  self.data.shape[0]*self.dt,  self.dt)
    #    print (np.max(self.timebase), self.dt)

        # Weiner filtering
        starttime = timeit.default_timer()
        H = np.fft.fft(self.template)
        if H.shape[0] < self.data.shape[0]:
            H = np.hstack((H,  np.zeros(self.data.shape[0]-H.shape[0])))
        self.quot = np.fft.ifft(np.fft.fft(self.data)*np.conj(H)/(H*np.conj(H) + llambda**2.0))
        self.Crit = np.real(self.quot)
        sd = np.std(self.Crit)
        self.sdthr = sd * thresh  # set the threshold
        self.above = np.clip(self.Crit,  self.sdthr,  None)
        self.onsets = scipy.signal.argrelextrema(self.above,  np.greater,  order=int(order))[0] - 1 + self.idelay
        self.summarize(data)
        endtime = timeit.default_timer() - starttime
        print('AJ run time: {0:.4f} s'.format(endtime))


#  Some general functions

def moving_average(a,  n=3) :
    ret = np.cumsum(a,  dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n, n


def generate_testdata(dt,  maxt=1e4, meanrate=10,  amp=20.,  ampvar=5.,  
        noise=2.5, taus=[1.0, 10.0], func=None, sign=1, expseed=None, noiseseed=None):

    tdur = 100.
    timebase = np.arange(0.,  maxt,  dt) # in ms
    t_psc = np.arange(0.,  tdur,  dt)  # time base for single event template in ms
    if func is None:
        tau_1 = taus[0] # ms
        tau_2 = taus[1] # ms
        Apeak = amp # pA
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        g = Apeak/Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
    else:
        func._make_template()
        g = amp*func.template
    g = g * sign  # negative going events if sign neegative

        
    testpsc = np.zeros(timebase.shape)
    if expseed is None:
        eventintervals = np.random.exponential(1e3/meanrate, int(maxt))
    else:
        np.random.seed(expseed)
        eventintervals = np.random.exponential(1e3/meanrate, int(maxt))
        
    events = np.cumsum(eventintervals)
    t_events = events[events < maxt]  # time of events with exp distribution
    i_events = np.array([int(x/dt) for x in t_events])
    testpsc[i_events] = np.random.normal(1.,  ampvar/amp,  len(i_events))
    i_events = i_events-int((tdur)/dt)
    testpsc = scipy.signal.convolve(testpsc,  g,  mode='same')
    if noise > 0:
        if noiseseed is None:
            testpscn = testpsc + np.random.normal(0.,  noise,  testpsc.shape)
        else:
            np.random.seed(noiseseed)
            testpscn = testpsc + np.random.normal(0.,  noise,  testpsc.shape)
    else:
        testpscn = testpsc
    return timebase,  testpsc,  testpscn,  i_events


def cb_tests():
    """
    Do some tests of the CB protocol and plot
    """
    sign = -1
    trace_dur = 1e4
    dt = 0.1
    taus = [1., 5.]
    for i in range(10):
        timebase,  testpsc,  testpscn,  i_events = generate_testdata(dt, maxt=trace_dur,
            amp=20.,  ampvar=10.,  noise=5.0, taus=[1., 5.], func=None, sign=sign,
            expseed=i, noiseseed=i*47)
        cb = ClementsBekkers()
        cb.setup(tau1=1.,  tau2=5.,  dt=dt,  delay=0.0, template_tmax=3*taus[1],  sign=sign)

        cb.cbTemplateMatch(testpscn,  threshold=2.0)
    cb.plots()
    return cb


def aj_tests():
    sign = -1
    trace_dur = 1e4
    dt = 0.1
    for i in range(10):
        aj = AndradeJonas()
        aj.setup(tau1=1.,  tau2=5.,  dt=dt,  delay=0.0, template_tmax=trace_dur,  sign=sign)
        # generate test data
        timebase,  testpsc,  testpscn,  i_events = generate_testdata(aj.dt, maxt=trace_dur,
                amp=20.,  ampvar=10.,  noise=5.0, taus=[1., 5.], func=None, sign=sign,
                expseed=i, noiseseed=i*47)
        print(int(1./aj.dt))
        aj.deconvolve(testpscn-np.mean(testpscn),  thresh=2.0, llambda=7,  order=int(1.0/aj.dt))
    aj.plots(events=None) # i_events)
    return aj
    

if __name__ == "__main__":
    #aj = aj_tests()
    cb_tests()