import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import get_window, butter, lfilter
sys.path.append("../../../../software/models")
import utilFunctions as UF
import stft as STFT



timeLength = 3                     # seconds
fs = 44100
t = np.arange(0, timeLength, 1.0/fs)
ww = get_window("hamming", t.size)


# Chirps -----------------------------------------

A = np.e**-(t/4)                                    # Or np.arange(.8, .0, -.8/t.size)
f0 = 220.0                                     # Hz   
f1 = 512.0
k = np.power((f1/f0),1/timeLength) 
phi0 = np.pi / 2                                # phase, radiants.                	                               # sampling rate
s0 = A * np.sin(phi0 + 2*np.pi*f0*((np.power(k,t)-1)/np.log(k)))       # time function, sinusoid.

f0 = 440.0                                     # Hz   
f1 = 1024.0
k = np.power((f1/f0),1/timeLength)                              # phase, radiants.                	                               # sampling rate
s1 = A/32 * np.sin(phi0 + 2*np.pi*f0*((np.power(k,t)-1)/np.log(k)))       # time function, sinusoid.


f0 = 660.0                                     # Hz   
f1 = 2048.0
k = np.power((f1/f0),1/timeLength)                                # phase, radiants.                	                               # sampling rate
s2 = A/64* np.sin(phi0 + 2*np.pi*f0*((np.power(k,t)-1)/np.log(k)))       # time function, sinusoid.





# Nessuna finestra applicata all'output. 
# Sinusoide costante, rumore filtrato "dinamicamente"
ss = (s0+s1+s2)/3
zs = ss*ww

# PLOT AND SAVE

plt.figure(1, figsize=(15, 3))
plt.plot(t, zs, 'b', lw=1.5)
plt.axis([0, timeLength,-1.0,1.0])
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')

plt.tight_layout()
plt.savefig('synthesis2.png')
plt.show()

# STFT Analysis

Hop = 1024
M = 2048
N = 2*M
maxplotfreq = 4000.0
w = get_window('hann', M)
mX,pX = STFT.stftAnal(zs, w, N, Hop)
sizeEnv = int(mX[0,:].size)
binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv
numFrames = int(mX[:,0].size)
frmTime = Hop*np.arange(numFrames)/float(fs)                            
plt.pcolormesh(frmTime, binFreq,
np.transpose(mX[:,:int(sizeEnv*maxplotfreq/(.5*fs)+1)]))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('spectrogram2.png')
plt.show()

# Writing File

UF.wavwrite(zs, fs, "output2.wav")