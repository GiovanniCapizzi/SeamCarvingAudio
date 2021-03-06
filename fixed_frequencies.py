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
                                                             #      <--     [i-1, j]     -->
                                                            #        ..\        |        /..   
            for t in range(-hwidth, hwidth+1):              #                 [i,j]    
                if j+t>=0 and j+t<width:
                    distances[t+hwidth]=(distTo[i - 1][j + t] + mX[i][j])
    
            k = np.argmax(distances)             # choose the maximum energy in the previous row
            if distances[k] > distTo[i][j]:                           
                distTo[i][j] = distances[k]
                upLink[i][j] = k - hwidth        # shift k in range(-hwidth, hwidth+1)

    seam = [0] * height                          # y indexes of the found seam in mX
    seam[-1] = np.argmax(distTo[-1])             # starting with the maximum in the last row
    for i in range(height - 2, -1, -1):          # backtracking from the second last row (height-2) to 0
        seam[i] = seam[i + 1] + upLink[i + 1][seam[i + 1]] # calculate y (column) for each x (row) 

    return list(enumerate(seam))                 # [(x0,y0), (x1,y1), (x2,y2), ...]                                  


def strip(seam:list, mX:np.ndarray)->tuple:
    """ Remove the silence at the beginning of the seam and cut the right side 
        if there is a 60db of attenuation from its maximum magnitude value. 
        Returns the modified seam and its magnitude values.
         - seam: is a list of position in mX;
         - mX: the input spectrogram in linear scale and with dftFrames as rows.


        Note:
            To find the point where there is for example a 40dB of attenuation
            from the maximum magnitude (1% from its peak): 
            e.g peak = -31 (0.28 in linear scale) and attenuation= -40dB (0.01 in linear scale)
            20*log10(0.028) + 20*log10(0.01) = 20*log10(0.028*0.01) = -71dB 
        """

    magpts = np.array([mX[i][j] for i, j in seam])
    
    peak_index = np.argmax(magpts)                   # the index of the maximum value in the seam                                                                                                      
    if magpts[peak_index]<10**(-80/20):
        return None, None

    t = 10**(-80/20)                                 # silence threshold (in linear scale) for the left side of the seam 
    left_index = np.argmax(magpts>=t)
     
    stop =  magpts[peak_index] * 10**(-60/20)        # stop condition where the magnitude is below the 60db of attenuation
    right_index = len(seam)

    for i,value in enumerate(magpts[peak_index:]):                                     
        if value <= stop:                            # cut from maximum peak to the -60dB attenuation point
            right_index = peak_index+i                                  
            break

    seam = seam[left_index:right_index]
    magpts = magpts[left_index:right_index]

    return seam, magpts


def backward_integration(seam:list, mX:np.ndarray)->tuple:
    """
    Returns the decay in seconds obtained with Schroeder's backward integration 
     - seam: seam: is a list of position in mX;
     - mX: the input spectrogram in linear scale and with dftFrames as rows.
    """

    short_seam, magpts = strip(seam, mX)                # magpts: magnitude values for each seam point 
   
    if short_seam is None:
        return 0,0,0
                                                    
    bwint = np.zeros(len(magpts))                       # prepare the integration array
    bwint[::-1] = np.cumsum(magpts[::-1])               # backward integration over magnitude values  
    y = 20 * np.log10(bwint)                            # converts result in dB

                                                        # linear regression:

    X = np.array([np.ones(len(y)), np.arange(len(y))])  # the first row to zeros, to allow to allow a 
                                                        # non-zero intercept in the line equation
                                                        # [1,1,1,1,1,...]
                                                        # [0,1,2,3,4,...] <- x = np.arange(y)
                                        
                                                        # The np.matmul implements the @ operator
                                                        # calculate the (Moore-Penrose) pseudo-inverse 
    w = np.dot(pinv(X @ X.T) @ X, y.T)                  # using the normal equation
    Xt = np.dot(w.T, X)                                 # obtaining the new y elements from the line equation

    attenuated_curve = np.abs(Xt - Xt[0] + 60)                # searching the intersection with x => time           
    return np.argmin(attenuated_curve), np.array(short_seam)  # seam has been stripped from left to right      
  


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
    decay, short_seam = backward_integration(seam, mX)      
                                                                                                                   
    if not decay:
        return (0,)*4                                                                                           

    if debug:                                                # shows the seam on the spectrogram
        tmX = np.full(mX.shape, np.nan)                      # mX like matrix only for plot purpose
        for i, j in short_seam:
            tmX[i][j] = 1.0
        TS.stft_plot(mX, N, H, fs, show=False)
        TS.stft_plot(tmX, N, H, fs, show=True, mask=True)
    
 
    freq_bin, freq_mag, freq_phase = UF.peakInterp(mX[argPeakMax],pX[argPeakMax],seam[argPeakMax][1])


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

    return 20*np.log10(freq_mag), decay * H / fs , freq_bin * fs / N, freq_phase     # convert and returns all values



if __name__ == '__main__':                          

    M = 4096                                             # window size, zero-phase-windowing if M<N
    N = 4096                                             # fft size, must be a power of two
    H = M//4                                             # hopsize, 1/4 of M
    auto = False                                         # compute the magnitude and phase of a generated sound (optional)

    inputFile = './campana.wav'
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

    print("frequency; decay; magnitude; phase;\n")
    for i in range(7):
        peak, decay, freq, phase = popMode(mX, pX, N, H, fs, 0)
        if not freq:
            print("[{0}] Seam not available".format(i))
            continue                  
        print('[f] = {0:.0f}Hz;\t{1:.3f}s; {2:.3f}dB; {3:2f}\n'.format(freq, decay, peak-mdB, phase))

