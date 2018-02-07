from __future__ import print_function

"""
Classes for methods that do analysis of miniature synaptic potentials

Current implementations are ClementsBekkers and AndradeJonas

Test runs:
cb: 0.175 s (with cython version of algorithm); misses overlapping events
aj: 0.028 s, plus gets overlapping events

July 2017 Paul B. Manis

"""
import numpy as np
import scipy.signal
import matplotlib.pyplot as mpl
import digital_filters as dfilt
import timeit
from scipy.optimize import curve_fit
from numba import jit
import clembek

# @jit(nopython=True,  cache=True)
# def nb_clementsbekkers(data,  template):
#     ## Prepare a bunch of arrays we'll need later
#     n_template = len(template)
#     n_data = data.shape[0]
#     n_dt = n_data - n_template
#     sum_template = template.sum()
#     sum_template_2 = (template*template).sum()
#
#     data_2 = data*data
#     sum_data = np.sum(data[:n_template])
#     sum_data_2 = data_2[:n_template].sum()
#     scale = np.zeros(n_dt)
#     offset = np.zeros(n_dt)
#     crit = np.zeros(n_dt)
#     for i in range(n_dt):
#         if i > 0:
#             sum_data = sum_data + data[i+n_template] - data[i-1]
#             sum_data_2 = sum_data_2 + data_2[i+n_template] - data_2[i-1]
#         sum_data_template_prod = np.multiply(data[i:i+n_template],  template).sum()
#         scale[i] = (
#                     (sum_data_template_prod - sum_data * sum_template/n_template)/
#                     (sum_template_2 - sum_template*sum_template/n_template)
#                      )
#         offset[i] = (sum_data - scale[i]*sum_template)/n_template
#         fitted_template = template * scale[i] + offset[i]
#         sse = ((data[i:i+n_template] - fitted_template)**2).sum()
#         crit[i] = scale[i]/np.sqrt(sse/(n_template-1))
#     DC = scale/ crit
#     return(DC,  scale,  crit)


class ClementsBekkers():
    def __init__(self):
        self.dt = None
        self.data = None
        self.template = None

    def make_template(self,  tau_1,  tau_2,  tmax,  dt):
        """
        Makes a template for the sliding scale match.
        SImple double exponential function,  scaled to a max of 1.0
        
        Parameters
        ----------
        tau_1 : float (no default)
            time constant for rising exponential
        tau_2 : float (no default)
            time constant for decaying exponential
        tmax : float (no default)
            duration of the template
        dt : float (no default)
            sample rate for the template 
        """
        
        self.dt = dt
        t_psc = np.arange(0,  tmax,  dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        g = 1./Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
        self.template = g

    def clements_bekkers_cython(self,  data):
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
        return(offset,  scale,  crit)
        
    def clements_bekkers(self,  data):
        """
        Implements Clements-bekkers algorithm: slides template across data,          returns array of points indicating goodness of fit.
        Biophysical Journal,  73: 220-229,  1997.
        
        Parameters
        ----------
        data : np.array (no default)
            1D data array
        
        """
    
        ## Strip out meta-data for faster computation
        D = data.view(np.ndarray)
        T = self.template.view(np.ndarray)
        starttime = timeit.default_timer()
        #DC,  S,  crit = nb_clementsbekkers(D,  T)
        DC,  S,  crit = self.clements_bekkers_cython(D)
        endtime = timeit.default_timer() - starttime
        print('CB run time: %f s',  endtime)
        self.DC = DC
        self.Scale = S
        self.Crit = self.sign*crit  # assure that crit is positive
    
    def cbTemplateMatch(self,  data,  template=None,  threshold=3.0,  sign=1):
        self.data = data
        self.sign = sign
        self.threshold = threshold
        if template is not None:
            self.template = template
        self.clements_bekkers(sign*data)  # flip data sign if necessary
        self.Crit = sign*self.Crit
        mask = self.Crit > threshold
        diff = mask[1:] - mask[:-1]
        times = np.argwhere(diff==1)[:,  0]  ## every time we start OR stop an event
    
        ## in the unlikely event that the very first or last point is matched,  remove it
        if abs(self.Crit[0]) > threshold:
            times = times[1:]
        if abs(self.Crit[-1]) > threshold:
            times = times[:-1]
    
        nEvents = len(times) / 2
        result = np.empty(nEvents,  dtype=[('onset_indx',  int),  ('peak_indx',  int),  ('dc',  float),  ('scale',  float),  ('crit',  float)])
        i = 0
        for j in range(nEvents):
            i1 = times[i]
            i2 = times[i+1]
            d = self.Crit[i1:i2]
            p = np.argmax(d)
            pk = np.argmax(sign*self.data[i1:i2]) + i1
            # if int(pk) < p+i1:
            #     print( p,  i1,  i2,  pk)
            result['onset_indx'][j] = p+i1
            result['peak_indx'][j] = int(pk)
            result['dc'][j] = d[p]
            result['scale'][j] = self.Scale[p+i1]
            result['crit'][j] = self.Crit[p+i1]
            i = i + 2
        self.result = result
        return result

    def plots(self):
        fig,  ax = mpl.subplots(4,  1)
        for i in range(1,  len(ax)):
            ax[i].get_shared_x_axes().join(ax[i],  ax[0])
            ax[i].set_xticklabels([])
        tb = np.arange(0.,  len(self.data)*self.dt,  self.dt)
        ax[0].plot(tb,  self.data,  'k-',  linewidth=0.33)
        ax[0].set_title('data')
        ax[1].plot(tb[:self.DC.shape[0]],  self.DC)
        ax[1].set_title('DC')
        ax[2].plot(tb[:self.Scale.shape[0]],  self.Scale)
        ax[2].set_title('Scale')
        ax[3].plot(tb[:self.Crit.shape[0]],  self.Crit)
        ax[3].plot(tb[:self.Crit.shape[0]],  self.threshold*np.ones_like(tb[:self.Crit.shape[0]]),  'r--')
        ax[3].set_title('Crit')
        mdat = self.data.copy()
        cx = np.where(self.Crit < self.threshold)
        mdat[cx] = np.nan
        ax[0].plot(tb,  mdat,  'r--',  linewidth = 0.75)
        # decorate traces with markers for onsets and peaks
        for j in range(len(self.result)):
            k0 = self.result['onset_indx'][j]
            k1 = self.result['peak_indx'][j]
            ax[0].plot(tb[k0],  self.data[k0],  'ko',  markersize=3)
            ax[0].plot(tb[k1],  self.data[k1],  'y^',  markersize=3)
        ax[3].set_ylim([0.,  self.threshold*20])
        mpl.show()


class AndradeJonas(object):
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
        self.tmax = None
        self.delay = 0.
        self.template_max = 0.

    def setup(self,  tau1=None,  tau2=None,  template_tmax=None,  dt=None,  delay=0.0,  sign=1):
        """
        Just store the parameters - will compute when needed"""
        assert sign in [-1, 1]
        self.sign = sign
        self.taus = [tau1,  tau2]
        self.dt = dt
        self.tmax = template_tmax
        self.delay = int(delay/dt)  # points delay in template with zeros
        self.template = None  # reset the template if needed.
        
    def _make_template(self):
        """
        Private function: make template when it is needed
        """
        tau_1,  tau_2 = self.taus
        t_psc = np.arange(0,  self.tmax,  self.dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        self.template = np.zeros_like(t_psc)
        tm = 1./Aprime * ((1-(np.exp(-t_psc/tau_1)))**4 * np.exp((-t_psc/tau_2)))
        if self.delay > 0:
            self.template[self.delay:] = tm[:-self.delay]  # shift the template
        else:
            self.template = tm
        self.template_max = np.max(self.template)
        if self.sign == -1:
            self.template = -self.template
        
    def deconvolve(self,  data,  thresh=1,  llambda=5.0,  order=7,  events=None):
        if self.template is None:
            self._make_template()
        #self.data = self.sign*dfilt.SignalFilter_LPFButter(data,  2000.,  1000./self.dt,  NPole=8)
        self.data = data
        self.timebase = np.arange(0.,  self.data.shape[0]*self.dt,  self.dt)
        # Weiner filtering
        starttime = timeit.default_timer()
        H = np.fft.fft(self.template)
        if H.shape[0] < self.data.shape[0]:
            H = np.hstack((H,  np.zeros(self.data.shape[0]-H.shape[0])))
        self.quot = np.fft.ifft(np.fft.fft(self.data)*np.conj(H)/(H*np.conj(H) + llambda**2))
        self.quot = np.real(self.quot)
        sd = np.std(self.quot)
        self.sdthr = sd * thresh  # set the threshold
        # threshold (all values below sdthr become 0)
        # scipy.stats.threshold was deprecated (no reason?) in 0.17; we run in 0.19
#        self.above = scipy.stats.threshold(self.quot,  self.sdthr)
        self.above = np.clip(self.quot,  self.sdthr,  None)
        self.onsets = scipy.signal.argrelextrema(self.above,  np.greater,  order=order)[0] - 1
        self.summarize()
        endtime = timeit.default_timer() - starttime
        #print('AJ run time: %f s',  endtime)

    def summarize(self,  order=11):
        """
        compute intervals,  peaks and ampitudes for all found events
        """
        self.intervals = np.diff(self.timebase[self.onsets])  # event intervals
        i_decay_pts = int(self.taus[1]/self.dt)  # decay window time (points)
        self.peaks = []
        self.smoothed_peaks = []
        self.amplitudes = []
        ndata = len(self.data)
        avgwin = int(1.0/self.dt)  # 1 msec averaging window for peak detection
        for j in range(len(self.onsets)):  # for every event
            i_end = i_decay_pts + self.onsets[j]  # distance from peak to end
            if i_end > ndata:  # keep within the array limits
                i_end = ndata
            if j < len(self.onsets)-1:
                if i_end > self.onsets[j+1]:
                    i_end = self.onsets[j+1]-1  # only go to next event start
            i_decay_n = len(self.data[self.onsets[j]:i_end])
            if i_decay_n < avgwin:
                avgwin = i_decay_n
            move_avg = moving_average(self.data[self.onsets[j]:i_end], n=avgwin)
            p = np.argmax(self.sign*move_avg)  # find peak of smoothed data
            #p =  scipy.signal.argrelextrema(self.sign*self.data[self.onsets[j]:
            #                     uwin],  np.greater,  order=order)[0]
            self.peaks.extend([int(p+self.onsets[j])])  # raw peak
            self.smoothed_peaks.extend([move_avg[p]])  # smoothed peak
            abase = np.mean(self.data[self.onsets[j]-10:self.onsets[j]-3])
            apeak = np.mean(self.data[self.peaks[-1]-3:self.peaks[-1]+3])
            amp = self.sign*apeak - self.sign*abase
            self.amplitudes.extend([amp])
        self.average_events()
#        self.fit_average_event()

    def average_events(self):
        # compute average event with length of template
        tdur = np.max((np.max(self.taus)*3.0, 5.0))  # go 3 taus or 5 ms past event
        tpre = 5.
        self.tpre = tpre
        self.avgnpts = int((tpre+tdur)/self.dt)  # points for the average
        npre = int(tpre/self.dt) # points for the pre time
        npost = int(tdur/self.dt)
        avg = np.zeros(self.avgnpts)
        self.allevents = np.zeros((len(self.onsets),  self.avgnpts))
#        print ('allevent shape: ', self.allevents.shape)
        k = 0
#        print('pre, post: ', npre, npost)
        for j, i in enumerate(self.onsets):
            if (i + npost) < len(self.data) and (i - npre) >= 0:
#                print('datashape: ', self.data[i-npre:i+npost].shape)
                self.allevents[k,:] = self.data[(i-npre+1):(i+npost+1)]
                k = k + 1
        self.allevents = self.allevents[0:k, :]  # trim unused
#        self.avgeventtb = np.arange(-tpre,  tdur-self.dt,  self.dt)
        self.avgevent = self.allevents.mean(axis=0)
#        print('average event shape: ', self.avgevent.shape)
        self.avgeventtb = np.arange(self.avgevent.shape[0])*self.dt
#        print('average event tb shape: ', self.avgeventtb.shape)

    def doubleexp(self, x, t, y, risepower=4.0):
        tm = x[0] * (1.0 - np.exp(-t/x[1]))**risepower * np.exp((-t/x[2]))
        return tm-y
    
    def fit_average_event(self):
        tsel = np.argwhere(self.avgeventtb > self.tpre)[0]  # only fit data in event,  not baseline
#        print('tsel: ', tsel)
        tsel = np.min(tsel)
        self.fittsel = tsel
        init_vals = [-10.,  0.5,  4.]
        bounds  = [(-4000., 0.075, 0.2), (4000., 10., 50.)]
#        print('tb: ', self.avgeventtb[tsel:])
#        print('vals: ', self.avgevent[tsel:])
        res = scipy.optimize.least_squares(self.doubleexp, init_vals,
                        bounds=bounds, args=(self.avgeventtb[tsel:]-self.tpre, self.avgevent[tsel:]))
#                                    self.avgevent[tsel:],  p0=init_vals, maxfev=5000)
#        print ('best vals: ',  best_vals)
        best_vals = res.x
        self.fitresult = best_vals
        self.best_fit = self.doubleexp(best_vals, self.avgeventtb[tsel:]-self.tpre,
            np.zeros_like(self.avgeventtb[tsel:]))
        # lmfit version - fails for odd reason
        # dexpmodel = Model(self.doubleexp)
        # params = dexpmodel.make_params(A=-10.,  tau_1=0.5,  tau_2=4.0,  dc=0.)
        # self.fitresult = dexpmodel.fit(self.avgevent[tsel:],  params,  x=self.avgeventtb[tsel:])
        #print(self.fitresult.fit_report())
        # print('init vals: ', init_vals)
        # print(' best vals: ', best_vals)
        self.Amplitude = best_vals[0]
        self.tau1 = best_vals[1]
        self.tau2 = best_vals[2]
        self.DC = 0. # best_vals[3]
        self.tsel = tsel
        # mpl.plot(self.avgeventtb[tsel:], self.avgevent[tsel:], 'k-')
        # mpl.plot(self.avgeventtb[tsel:], self.best_fit, 'r--')
        # mpl.show()

    def plots(self,  events=None,  title=None):
        data = self.data
        fig,  ax = mpl.subplots(3,  1)
        for i in range(1,2):
            ax[i].get_shared_x_axes().join(ax[i],  ax[0])
        tb = self.timebase[:len(data)]
        ax[0].plot(tb,  data,  'k-',  linewidth=0.75)  # original data
        ax[0].plot(tb[self.onsets],  data[self.onsets],  'k^',  
                        markersize=6,  markerfacecolor=(1,  1,  0,  0.8),  )
        if events is not None:
            ax[0].plot(tb[events],  data[events],  'go',  markersize=5)
#        ax[0].plot(tb[self.peaks],  self.data[self.peaks],  'r^')
        ax[0].plot(tb[self.peaks],  self.smoothed_peaks,  'r^')
        
        ax[1].plot(tb[:self.quot.shape[0]],  self.quot)  # deconvolution
        ax[1].plot([tb[0],tb[-1]],  [self.sdthr,  self.sdthr],  'r--',  linewidth=0.75)
        ax[1].plot(tb[self.onsets]+self.delay,  self.quot[self.onsets],  'y^')  # add delay to show event onsets correctly
        if events is not None:  # original events
            ax[1].plot(tb[:self.quot.shape[0]][events],  self.quot[events],
                    'ro',  markersize=5.)
        ax[2].plot(self.avgeventtb[:len(self.avgevent)],  self.avgevent)
        maxa = np.max(self.sign*self.avgevent)
        ax[2].plot(self.avgeventtb+self.tpre,  self.template[0:self.avgnpts]*maxa/self.template_max,  'r-')
#        ax[2].plot(self.avgeventtb[self.fittsel:],  self.best_fit,  'g--')
        if title is not None:
            fig.suptitle(title)
        mpl.show()

def moving_average(a,  n=3) :
    ret = np.cumsum(a,  dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def generate_testdata(dt,  meanrate=10,  amp=20.,  ampvar=5.,  noise=2.5):
    maxt = 1e4
    mean_rate = meanrate # Hz
    tdur = 100.
    timebase = np.arange(0.,  maxt,  dt) # in ms
    t_psc = np.arange(0.,  tdur,  dt)  # in ms
    tau_1 = 1.0 # ms
    tau_2 = 10.0 # ms
    Apeak = amp # pA
    Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
    g = Apeak/Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
    testpsc = np.zeros(timebase.shape)
    eventintervals = np.random.exponential(1e3/mean_rate,  1000)
    events = np.cumsum(eventintervals)
    t_events = events[events < maxt]  # time of events with exp distribution
    i_events = np.array([int(x/dt) for x in t_events])
    testpsc[i_events] = np.random.normal(1.,  ampvar/amp,  len(i_events))
    i_events = i_events-int((tdur/2.)/dt)
    testpsc = scipy.signal.convolve(testpsc,  g,  mode='same')
    if noise > 0:
        testpscn = testpsc + np.random.normal(0.,  noise,  testpsc.shape)
    else:
        testpscn = testpsc
    return timebase,  testpsc,  testpscn,  i_events


def cb_tests():
    """
    Do some tests of the CB protocol and plot
    """
    dt = 0.1
    timebase,  testpsc,  testpscn,  ievents = generate_testdata(dt,  amp=20.,  ampvar=5.,  noise=5.)
    cb = ClementsBekkers()
    cb.make_template(1,  10.,  100.,  0.1)
    sign = 1
    #cb.clements_bekkers(testpscn)
    cb.cbTemplateMatch(sign*testpscn,  threshold=2.,  sign=sign)
    cb.plots()
    return cb

def aj_tests():
    sign = 1
    dt = 0.1
    timebase,  testpsc,  testpscn,  i_events = generate_testdata(dt,  amp=20.,  ampvar=10.,  noise=5.0)
    aj = AndradeJonas()
    aj.setup(tau1=1.2,  tau2=10.,  dt=dt,  template_tmax=np.max(timebase),  sign=sign)
    aj.deconvolve(testpscn-np.mean(testpscn),  thresh=3.0,  events=i_events,  llambda=5.,  order=int(1./aj.dt))
    aj.plots(events=None) # i_events)
    return aj
    

if __name__ == "__main__":
    aj = aj_tests()
    #cb_tests()