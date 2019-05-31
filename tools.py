import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import sys
from scipy.signal import butter, lfilter, freqz
sys.path.append("../../../../software/models")
import utilFunctions as UF


def stft_plot(mX:np.ndarray, N : int = 4096 , H: int = 1024, fs :int=44100, show:bool=True, dB:bool=True, mask:bool=False, title:str=None):
    """
    Plot the input spectrogram with matplotlib 
    :param mX: spectrogram in linear scale and NOT transposed
    :param N: fftsize
    :param H: hopsize
    :param fs: frequency sample
    :param show: shows the plot
    :param dB: convert to decibel scale, default True
    :param title: title string for the plot, None for default
    :return: None
    """
    maxplotfreq = 8500.0                                    # frequency range to plot, in Hz

    if dB:
        mX = 20*np.log10(mX)                             

    numFrames = int(mX[:,0].size)                         
    frmTime = H*np.arange(numFrames)/float(fs)              # convert x in seconds
    binFreq = fs*np.arange(N*maxplotfreq/fs)/N              # convert y in frequencies
    if mask:
        # https://matplotlib.org/examples/pylab_examples/custom_cmap.html
        # create a colormap, cmap_name, colors, N=n_bin                                                                                 
        highlight = LinearSegmentedColormap.from_list('highlight', ['r', 'r'], N=1)          
        plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(N*maxplotfreq/fs+1)]), cmap=highlight)  
    else:
        plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(N*maxplotfreq/fs+1)]))
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.title(title if title else 'magnitude spectrogram')
    plt.autoscale(tight=True)
    if show:
        plt.show()

 
def dft_plot(mX:np.ndarray, color:str = 'r', N:int=4096, fs:int=44100, show:bool=True, dB:bool = False):
    """
    Plot the input spectrum with matplotlib 
    :param mX: spectrum in linear scale
    :param color: matplotlib color for the plot
    :param N: fftsize
    :param fs: frequency sample
    :param show: shows the plot
    :param dB: convert to dB scale
    :return: None
    """
    if dB:
        mX = 20*np.log10(mX)                                               # convert mX to dB scale
    binFreq = float(fs)*np.arange(mX.size)/float(N)                        # convert x in frequencies
    plt.plot(binFreq, mX, color)
    plt.axis([0, fs/2.0, np.min(mX), np.max(mX)])
    plt.title ('magnitude spectrum: mX')
    plt.ylabel('amplitude ({0})'.format('dB' if dB else 'linear scale'))
    plt.xlabel('frequency (Hz)')
    plt.autoscale(tight=True)
    if show:
        plt.show()

def bw_plot(cf: float, fs:int=44100):
    """
    Plots the butterwort filter with given its cutoff frequency
        :params cf: cutoff frequency in Hz for the filter, point at which the gain drops to 1/sqrt(2)
        :params fs: frequency sample
        :returns: None

    ref -> https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    """
    # Sample rate and desired cutoff frequencies (in Hz).
    Wn = cf / (fs/2)
    order = 2

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    b, a = butter(order, Wn, 'low')
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


def sinesynth(freqs: np.ndarray, mag: int = 0, N: int = 4096, H: int = 1024, T: float = 1.0, fs: int = 44100) -> tuple:
    """
    Create a sound spectrum gived its frequencies and fft parameters
        :param freqs: numpy array of frequencies 
        :param mag: magnitude in dB, each dftFrame will substract 1/e to this value
        :param N: fftsize
        :param H: hopsize
        :param T: duration of the sound in seconds
        :param fs: frequency sample
        :return: magnitude spectrogram of the created sound using UF.genSpecSines() from SMS-tools
    """
    tol = 1e-14                                                      # threshold used to compute phase
    numFrames = int(T*fs/H)                                          # number of dftFrames in the output spectrogram
    alpha = 20*np.log10((1/np.e)) / (fs / H) * 2.05                  # the quantity of decrement (in dB) in each dftFrame
    hN = N//2+1                                                      # size of positive spectrum
    xmX = []                                                         
    xpX = []                                                         
    n_freq = freqs.shape[0]                                          
    phases = np.array([0] * n_freq)                                  # phase for every sine, assuming it is constant (0 to all)
    for i in range(numFrames):
        magnitudes = np.array([mag] * n_freq)                        # magnitude array in dB 
        X = UF.genSpecSines(freqs, magnitudes, phases, N, fs)        # using only the fftsize N, suppose window size W = N
        absX = abs(X[:hN])                                           # compute absolute value of positive side
        absX[absX<1e-06] = 1e-06                                     # handle log, you can use also np.finfo(float).eps
        mX = 20 * np.log10(absX)                                     # magnitude spectrum of positive frequencies in dB
        X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0                 # for phase calculation set to 0 the small values
        X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0                 # for phase calculation set to 0 the small values         
        pX = np.unwrap(np.angle(X[:hN]))                             # unwrapped phase spectrum of positive frequencies
        xmX.append(np.array(mX))                                     # append output to list
        xpX.append(np.array(pX))
        mag += alpha
    xmX = np.array(xmX)                                              
    xpX = np.array(xpX)                      
    return xmX, xpX

def butterworth(mX: np.ndarray, Wn: float = 661.5, debug: bool = False,  N: int = 4096, fs: int = 44100):
    """
    Apply a second order butterworth filter to each spectrum in the spectrogram.
    See also: docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter,
              docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html,
              https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html (for normalization of parameters)

        :param mX: spectrogram in linear scale and not transposed
        :param Wn: cutoff frequency (positive and in dB) for the filter, point at which the gain drops to 1/sqrt(2)
        :param debug: activate plots for debugging
        :param N: fftsize required by dft_plot 
        :param fs: frequency sample required by dft_plot
        :return: None
    """ 
    nyq = 0.5 * fs                                    # Use nyquist frequency to normalize Wn
    cutoff = Wn / nyq
    [b, a] = butter(2, cutoff, 'low')
    if debug:
        bw_plot(Wn, fs)
    for i in range(mX.shape[0]):
        frame = lfilter(b, a, mX[i]) * 1.8            # filter the spectrum with the lowpass filter,
        #frame = np.roll(frame, -1)                   # centering the result, due to latency of the applied filter,
        peaks = mX[i] - frame;                        # deriving peak, this will create negative areas,
        if debug:
            dft_plot(mX[i], 'b', N, fs, False)
            dft_plot(frame, 'r', N, fs, True)
        mX[i][peaks <= 1e-04] = 1e-04                 # cut off where there are no peaks, giving -120dB to each element <= 0
        if debug:
            dft_plot(mX[i], 'b', N, fs, True)


