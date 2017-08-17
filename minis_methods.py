"""
Classes for methods that do analysis of miniature synaptic potentials

Current implementations are ClementsBekkers and AndradeJonas

July 2017 Paul B. Manis

"""
import numpy as np
import scipy.signal
import numpy.random
import matplotlib.pyplot as mpl
import digital_filters as dfilt
from lmfit import Model
from scipy.optimize import curve_fit


class ClementsBekkers():
    def __init__(self):
        self.dt = None
        self.data = None
        self.template = None

    def make_template(self, tau_1, tau_2, tmax, dt):
        """
        Makes a template for the sliding scale match.
        SImple double exponential function, scaled to a max of 1.0
        
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
        t_psc = np.arange(0, tmax, dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        g = 1./Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
        self.template = g
            
    def clements_bekkers(self, data):
        """
        Implements Clements-bekkers algorithm: slides template across data,
        returns array of points indicating goodness of fit.
        Biophysical Journal, 73: 220-229, 1997.
        
        Parameters
        ----------
        data : np.array (no default)
            1D data array
        
        
        """
    
        ## Strip out meta-data for faster computation
        D = data.view(np.ndarray)
        T = self.template.view(np.ndarray)
        self.data = D
        
        ## Prepare a bunch of arrays we'll need later
        N = len(T)
        sumT = T.sum()
        sumT2 = (T**2.0).sum()
        D2 = D**2.0
        NDATA = len(data)
        crit = np.zeros(NDATA)

        sumD = np.zeros((NDATA-N))
        sumD2 = np.zeros((NDATA-N))
        sumDTprod = np.zeros((NDATA-N))
        sumD[0] = D[:N].sum()
        sumD2[0] = D2[:N].sum()
        # sumD = rollingSum(D[:N], N)
        # sumD2 = rollingSum(D2[:N], N)
        for i in range(NDATA-N):
            if i > 0:
                sumD[i] = sumD[i-1] + D[i+N] - D[i]
                sumD2[i] = sumD2[i-1] + D2[i+N] - D2[i]
            sumDTprod[i] = (D[i:N+i]*T).sum()
        S = (sumDTprod - sumD*sumT/N)/(sumT2 - sumT*sumT/N)
        C = (sumD - S*sumT)/N
        SSE = sumD2 + (S*S*sumT2) + (N*C*C) - 2.0*(S*sumDTprod + C*sumD - (S*C*sumT))
        crit = S/np.sqrt(SSE/(N-1))
        DC = S / crit
        self.DC = DC
        self.Scale = S
        self.Crit = crit

    def rollingSum(self, data, n):
        d1 = data.copy()
        d1[1:] = np.cumsum(d1[1:]) # d1[1:] + d1[:-1]  # integrate
        d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
        d2[0] = d1[n-1]  # copy first point
        d2[1:] = d1[n:] - d1[:-n]  # subtract
        return d2
    
    def clementsBekkers2(self, data):
        """Implements Clements-bekkers algorithm: slides template across data,
        returns array of points indicating goodness of fit.
        Biophysical Journal, 73: 220-229, 1997.
    
        Campagnola's version...
        """
    
        ## Strip out meta-data for faster computation
        D = data.view(np.ndarray)
        T = self.template.view(np.ndarray)
        NDATA = len(D)
        ## Prepare a bunch of arrays we'll need later
        N = len(T)
        sumT = T.sum()
        sumT2 = (T**2.).sum()
        sumD = self.rollingSum(D, N)
        sumD2 = self.rollingSum(D**2., N)
        sumTD = scipy.signal.correlate(D, T, mode='valid')
    
        ## compute scale factor, offset at each location:
        scale = (sumTD - sumT * sumD /N) / (sumT2 - sumT**2. /N)
        offset = (sumD - scale * sumT) /N
    
        ## compute SSE at every location
        SSE = sumD2 + scale**2.0 * sumT2 + N * offset**2. - 2. * (scale*sumTD + offset*sumD - scale*offset*sumT)
        ## finally, compute error and detection criterion
        error = np.sqrt(SSE / (N-1))
        DC = scale / error
        self.DC = DC
        self.Scale = scale
        self.Crit = error
    
    def cbTemplateMatch(self, data, threshold=3.0):
        self.clementsBekkers(data)
        mask = self.Crit > threshold
        diff = mask[1:] - mask[:-1]
        times = np.argwhere(diff==1)[:, 0]  ## every time we start OR stop a spike
    
        ## in the unlikely event that the very first or last point is matched, remove it
        if abs(self.Crit[0]) > threshold:
            times = times[1:]
        if abs(self.Crit[-1]) > threshold:
            times = times[:-1]
    
        nEvents = len(times) / 2
        result = np.empty(nEvents, dtype=[('peak', int), ('dc', float), ('scale', float), ('offset', float)])
        for i in range(nEvents):
            i1 = times[i*2]
            i2 = times[(i*2)+1]
            d = self.Crit[i1:i2]
            p = np.argmax(d)
            result[i][0] = p+i1
            result[i][1] = d[p]
            result[i][2] = self.Scale[p+i1]
            result[i][3] = self.Crit[p+i1]
        return result

    def plots(self):
        fig, ax = mpl.subplots(4, 1)
        tb = np.arange(0., len(self.data)*self.dt, self.dt)
        ax[0].plot(tb, self.data)
        ax[1].plot(tb[:self.DC.shape[0]], self.DC)
        ax[2].plot(tb[:self.Scale.shape[0]], self.Scale)
        ax[3].plot(tb[:self.Crit.shape[0]], self.Crit)
        cx = [self.Crit > 2.0]
        ax[0].plot(tb[cx], self.data[cx], 'r')
        mpl.show()


class AndradeJonas(object):
    """
    Deconvolution method of Andrade/Jonas, Biophysical Journal 2012
    
    """
    def __init__(self):
        self.template = None
        self.onsets = None
        self.timebase = None
        self.dt = None
        self.sign = 1
        self.taus = None
        self.tmax = None

    def setup(self, tau1=None, tau2=None, tmax=None, dt=None, delay=1.0, sign=1):
        """
        Just store the parameters - will compute when needed"""
        self.sign = sign
        self.taus = [tau1, tau2]
        self.dt = dt
        self.tmax = tmax
        self.delay = int(delay/dt)  # points delay in template with zeros
        self.template = None  # reset the template if needed.
        
    def _make_template(self):
        """
        Private function: make template when it is needed
        """
        tau_1, tau_2 = self.taus
        t_psc = np.arange(0, self.tmax, self.dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        self.template = np.zeros_like(t_psc)
        tm = 1./Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
        self.template[self.delay:] = tm[:-self.delay]  # shift the template
        self.template_max = np.max(self.template)
        if self.sign == -1:
            self.template = -self.template
        
    def deconvolve(self, data, thresh=1, llambda=5.0, order=7, events=None):
        if self.template is None:
            self._make_template()
        self.data = dfilt.SignalFilter_LPFButter(data, 2000., 1000./self.dt, NPole=8)
        # Weiner filtering
        self.timebase = np.arange(0., self.tmax+self.dt, self.dt)
        H = np.fft.fft(self.template)
        if H.shape[0] < self.data.shape[0]:
            H = np.hstack((H, np.zeros(self.data.shape[0]-H.shape[0])))
        self.quot = np.fft.ifft(np.fft.fft(data)*np.conj(H)/(H*np.conj(H) + llambda**2))
        self.quot = np.real(self.quot)
        sd = np.std(self.quot)
        self.sdthr = sd * thresh  # set the threshold
        # threshold (all values below sdthr become 0)
        self.above = scipy.stats.threshold(self.quot, self.sdthr)
        self.onsets = scipy.signal.argrelextrema(self.above, np.greater, order=order)[0] - 1
        self.summarize()

    def summarize(self, order=11):
        """
        compute intervals, peaks and ampitudes for all found events
        """
        self.intervals = np.diff(self.timebase[self.onsets])
        mwin = int(self.taus[1]/self.dt)
        self.peaks = []
        self.smoothed_peaks = []
        self.amplitudes = []
        ndata = len(self.data)
        for j in range(len(self.onsets)):
            avgwin = int(1.0/self.dt)
            uwin = mwin + self.onsets[j]
            if uwin > ndata:  # keep within the array limits
                uwin = ndata
            if j < len(self.onsets)-1:
                if uwin > self.onsets[j+1]:
                    uwin = self.onsets[j+1]-1  # only go to the next event start
            ntest = len(self.data[self.onsets[j]:uwin])
            if ntest < avgwin:
                avgwin = ntest
            m = moving_average(self.data[self.onsets[j]:uwin], n=avgwin)
            p = np.argmax(self.sign*m)
            #p =  scipy.signal.argrelextrema(self.sign*self.data[self.onsets[j]:
            #                     uwin], np.greater, order=order)[0]
            self.peaks.extend([int(p+self.onsets[j])])
            self.smoothed_peaks.extend([m[p]])
            abase = np.mean(self.data[self.onsets[j]-5:self.onsets[j]])
            apeak = np.mean(self.data[self.peaks[-1]-3:self.peaks[-1]+3])
            amp = self.sign*apeak - self.sign*abase
            self.amplitudes.extend([amp])
        self.average_events()
#        self.fit_average_event()

    def average_events(self):
        # compute average event with length of template
        tdur = 50.
        tpre = 5.
        self.avgnpts = int((tpre+tdur)/self.dt)+1
        npts = self.avgnpts
        npre = int(tpre/self.dt)
        avg = np.zeros(self.avgnpts)
#        print ('original: ', len(self.onsets), self.avgnpts)
        self.allevents = np.zeros((len(self.onsets), self.avgnpts))
        nev = 0  # count accepted events
        for j, i in enumerate(self.onsets):
            if (i + npts) < len(self.data) and (i - npre) >= 0:
                self.allevents[j,:] = self.data[i-npre:i+self.avgnpts-npre]
                avg = avg + self.data[i-npre:i+self.avgnpts-npre]
                nev = nev + 1
        if nev < len(self.onsets):
            self.allevents = self.allevents[0:nev, :]
#        print('final: ', self.allevents.shape)
        avg = avg/nev
        ttb = np.arange(-tpre, tdur+self.dt, self.dt)
        self.avgeventtb = ttb
        self.avgevent = avg
        self.tpre = tpre

    def doubleexp(self, x, A, tau_1, tau_2, dc):
        tm = A * (-np.exp(-x/tau_1) + np.exp((-x/tau_2))) + dc
        return tm
    
    def fit_average_event(self):
        tsel = np.argwhere(self.avgeventtb > self.tpre)  # only fit data in event, not baseline
        tsel = np.min(tsel)
        self.fittsel = tsel
        init_vals = [-10., 0.5, 4., 0.]
        best_vals, covar = curve_fit(self.doubleexp, self.avgeventtb[tsel:], self.avgevent[tsel:], p0=init_vals)
#        print ('best vals: ', best_vals)
        self.fitresult = best_vals
        self.best_fit = self.doubleexp(self.avgeventtb[tsel:], best_vals[0], best_vals[1], best_vals[2], best_vals[3])
        # lmfit version - fails for odd reason
        # dexpmodel = Model(self.doubleexp)
        # params = dexpmodel.make_params(A=-10., tau_1=0.5, tau_2=4.0, dc=0.)
        # self.fitresult = dexpmodel.fit(self.avgevent[tsel:], params, x=self.avgeventtb[tsel:])
        # print(self.fitresult.fit_report())
        self.tau1 = best_vals[1]
        self.tau2 = best_vals[2]

    def plots(self, events=None):
        data = self.data
        dt = self.dt
        fig, ax = mpl.subplots(3, 1)
        for i in range(1,2):
            ax[i].get_shared_x_axes().join(ax[i], ax[0])
#        print(self.timebase.shape[0], self.data.shape[0])
        tb = self.timebase
#        print ('len tb: ', len(tb), '  len data: ', len(data))
        ax[0].plot(tb, data, 'k-', linewidth=0.75)  # original data
        ax[0].plot(tb[self.onsets], data[self.onsets], 'k^', markersize=6, markerfacecolor=(1, 1, 0, 0.8), )
        if events is not None:
            ax[0].plot(tb[events], data[events], 'go', markersize=5)
#        ax[0].plot(tb[self.peaks], self.data[self.peaks], 'r^')
        ax[0].plot(tb[self.peaks], self.smoothed_peaks, 'r^')
        
        ax[1].plot(tb[:self.quot.shape[0]], self.quot)  # deconvolution
        ax[1].plot([tb[0],tb[-1]], [self.sdthr, self.sdthr], 'r--', linewidth=0.75)
        ax[1].plot(tb[self.onsets], self.quot[self.onsets], 'y^')
        if events is not None:  # original events
            ax[1].plot(tb[:self.quot.shape[0]][events], self.quot[events], 'ro', markersize=5.)



        ax[2].plot(self.avgeventtb, self.avgevent)
        maxa = np.max(self.sign*self.avgevent)
        ax[2].plot(self.avgeventtb+self.tpre, self.template[0:self.avgnpts]*maxa/self.template_max, 'r-')
#        ax[2].plot(self.avgeventtb[self.fittsel:], self.best_fit, 'g--')
        mpl.show()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def generate_testdata(dt, meanrate=10, amp=20., ampvar=5., noise=2.5):
    maxt = 1e4
    mean_rate = meanrate # Hz
    tdur = 100.
    timebase = np.arange(0., maxt, dt) # in ms
    t_psc = np.arange(0., tdur, dt)  # in ms
    tau_1 = 1.0 # ms
    tau_2 = 10.0 # ms
    Apeak = amp # pA
    Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
    g = Apeak/Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
    testpsc = np.zeros(timebase.shape)
    eventintervals = np.random.exponential(1e3/mean_rate, 1000)
    events = np.cumsum(eventintervals)
    t_events = events[events < maxt]  # time of events with exp distribution
    i_events = np.array([int(x/dt) for x in t_events])
    testpsc[i_events] = np.random.normal(1., ampvar/amp, len(i_events))
    i_events = i_events-int((tdur/2.)/dt)
    testpsc = scipy.signal.convolve(testpsc, g, mode='same')
    if noise > 0:
        testpscn = testpsc + np.random.normal(0., noise, testpsc.shape)
    else:
        testpscn = testpsc
    return timebase, testpsc, testpscn, i_events


def cb_tests():
    """
    Do some tests of the CB protocol and plot
    """
    timebase, testpsc, testpscn, ievents = generate_testdata()
    cb = ClementsBekkers()
    cb.make_template(1, 10., 100., 0.1)
    cb.clements_bekkers(testpscn)
    cb.plots()

def aj_tests():
    dt = 0.1
    timebase, testpsc, testpscn, i_events = generate_testdata(dt, amp=20., ampvar=5., noise=5.0)
    aj = AndradeJonas()
    aj.setup(tau1=1.2, tau2=10., dt=dt, tmax=np.max(timebase))
    aj.deconvolve(testpscn-np.mean(testpscn), thresh=3.0, events=i_events, llambda=5., order=int(1./aj.dt))
    aj.plots(events=None) # i_events)
    

if __name__ == "__main__":
    aj_tests()
#    cb_tests()