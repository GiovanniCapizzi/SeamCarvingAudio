import sys
import numpy as np
sys.path.append("../../../software/models")

import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.linalg import pinv
import utilFunctions as UF
import stft as STFT
import tools as TS


def findVerticalSeam(mX:np.ndarray)->list:
    """ Returns a list of coordinates rappresenting a vertical seam found in the given spectrogram.
         - mX: the input spectrogram in linear scale and with dftFrames as rows. """

    height, width = mX.shape                                         
    distTo = np.full(mX.shape, -np.inf)                 # initialize the energy matrix to -inf values
    distTo[0,:] = 0                                     # initialize the first row to zeros
    upLink = np.full(mX.shape, 0)                       # initialize the backtracking matrix 
                         
    hwidth = 3                                          # adjacent span 
    span = hwidth*2+1
    for i in range(1, height):                          # update values from the second row to the last one
        for j in range(width):
            distances = [-np.inf]*span
                                                            #            <--     [i-1, j]     -->
                                                            #             ..\        |        /..   
            for t in range(-hwidth, hwidth+1):              #                      [i,j]    
                if j+t>=0 and j+t<width:
                    distances[t+hwidth]=(distTo[i - 1][j + t] + mX[i][j])
    
            k = np.argmax(distances)                    # choose the maximum energy in the previous row
            if distances[k] > distTo[i][j]:                           
                distTo[i][j] = distances[k]
                upLink[i][j] = k - hwidth               # shift k in range(-hwidth, hwidth+1)

    seam = [0] * height                                 # y indexes of the found seam in mX
    seam[-1] = np.argmax(distTo[-1])                    # starting with the maximum in the last row
    for i in range(height - 2, -1, -1):                    # backtracking from the second last row (height-2) to 0
        seam[i] = seam[i + 1] + upLink[i + 1][seam[i + 1]] # calculate y (column) for each x (row) 

    return list(enumerate(seam))                           # [(x0,y0), (x1,y1), (x2,y2), ...]                                

    
def lr_strip(seam:list, mX:np.ndarray)->tuple:
    """ Remove the silence at the beginning and at the end of the seam. 
        Returns the modified seam and its magnitude values.
         - seam: is a list of position in mX;
         - mX: the input spectrogram in linear scale and with dftFrames as rows.
        """

    magpts = np.array([mX[i][j] for i, j in seam])

    t = 10**(-80/20)                                 # silence threshold (in linear scale) for the left side of the seam 
    left_index = np.argmax(magpts>=t)
    right_index = np.argmax(np.flipud(magpts)>=t) - 1

    if left_index==len(magpts)-right_index:
        return None, None

    seam = seam[left_index:-right_index]
    magpts = magpts[left_index:-right_index]

    return seam, magpts


def popMode(mX:np.ndarray, pX:np.ndarray=None, N:int=4096, H:int=1024, fs:int=44100, debug:int=0)->tuple:
    """
    Returns the extracted parameters of first partial found in the spectrogram, 
    than removes it from mX.
     - mX: the input spectrogram in linear scale and with dftFrames as rows.
     - pX: phase spectrum, 0 angle phase if None.
     - N: fftsize
     - H: hopsample
     - fs: frequency sample
     - debug: shows the spectrograms plot
    """

    if pX is None:
        pX = np.full(mX.shape, 0)                           # phases are 0 if there is no pX in input

    seam = findVerticalSeam(mX)
    argPeakMax = np.argmax([mX[i][j] for i, j in seam])
    short_seam, magpts = lr_strip(seam, mX)   
                                                                                                                   
    if seam is None:
        return (0,)*4                                                                                           

    if debug:                                                # shows the seam on the spectrogram
        tmX = np.full(mX.shape, np.nan)                      # mX like matrix only for plot purpose
        for i, j in short_seam:
            tmX[i][j] = 1.0
        TS.stft_plot(mX, N, H, fs, show=False)
        TS.stft_plot(tmX, N, H, fs, show=True, mask=True)


    for i, j in short_seam:
        if mX[i][j]<1e-05:
            continue     

        f_bin, f_mag, f_phase = UF.peakInterp(mX[i],pX[i],j)

        if f_mag < 1e-05: 
            f_mag = 1e-05

        f_mag = 20 * np.log10(f_mag) + 2                            # convert magnitude to dB (genSpecSines expect dB magnitudes)
        f_bin = f_bin * fs / N                                      # convert y or j-th column to frequency

        y = UF.genSpecSines(f_bin, f_mag, pX[i][j], N, 44100)       # generate a spectrum frame for the input frequencies and phases
        y = abs(y)[:N//2 + 1]                                       # positive half of the spectrum
     
        mX[i] = mX[i] - y                                           # subtract the generated frequencies
        mX[i][mX[i] <= 1e-06] = 1e-06                               # giving 10**(-120dB/20) to each element <=0 in linear scale

 
    if debug == 1:
        TS.stft_plot(mX, N, H, fs, show=True)

    freqs = [j* fs / N for i,j in short_seam]
    phases = [pX[i][j] for i,j in short_seam]
    return 20*np.log10(magpts), len(seam) * H / fs , freqs, phases     # convert and returns all values



if __name__ == '__main__':                          

    M = 1024                                             # window size, zero-phase-windowing if M<N
    N = 1024                                             # fft size, must be a power of two
    H = M//4                                             # hopsize, 1/4 of M
    auto = False                                         # compute the magnitude and phase of a generated sound (optional)

    inputFile = './bouncing_sound.wav'
    window = 'blackmanharris'                            # window type

    
    fs, x = UF.wavread(inputFile)                        # read input sound (monophonic with sampling rate of 44100)   
    w = get_window(window, M)                            # compute analysis window

    
    if auto:
        mX, pX = TS.sinesynth(np.array([500,1300,1700]), -11, N, H, 4.363, fs)
    else:
        mX, pX = STFT.stftAnal(x, w, N, H)               # compute the magnitude and phase spectrogram: i -> rows, j -> columns
    
    mdB = np.max(mX)
    mX = 10 ** (mX / 20)                                 # convert the magnitude to linear scale

    # TS.butterworth(mX, 50, False, N, fs)          


    for i in range(7):
        peak, decay, freq, phase = popMode(mX, pX, N, H, fs, 1) 
        if not freq:
            print("[{0}] Seam not available".format(i))
            continue 
        print("[{0}] Seam has a decay of {1} seconds".format(i, decay))

        
        plt.subplot(3,1,1)
        plt.title('Magnitude')
        plt.plot(peak)
 
        plt.subplot(3,1,2)
        plt.title('freq')
        plt.plot(freq)
 
        plt.subplot(3,1,3)
        plt.title('phase')
        plt.plot(phase)
 
        plt.show()
        
