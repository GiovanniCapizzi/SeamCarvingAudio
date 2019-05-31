import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import get_window, butter, lfilter
sys.path.append("../../../software/models")
import utilFunctions as UF
import stft as STFT



timeLength = 4.0                        # seconds
fs = 44100
t = np.arange(0, timeLength, 1.0/fs)
ww = get_window("hann", t.size)


# CHIRP 1 ------------------------------------------------
Ar = .3                                 # amplitude
fi1 = 1000.0                              # initial frequency
ff1 = 500.0                              # final frequency
phi0 = 0                                
# linear frequency sweep
# eq. ->  https://en.wikipedia.org/wiki/Chirp
# with -> k = (ff-fi)/timeLength
phi = phi0 + 2*np.pi*fi1*t + np.pi*(ff1 - fi1)/timeLength*t*t
s1 = Ar * np.cos(phi)

# CHIRP 2 ------------------------------------------------
fi2 = 250.0                              # initial frequency
ff2 = 750.0                              # final frequency
phi0 = 0                                
# linear frequency sweep
phi = phi0 + 2*np.pi*fi2*t + np.pi*(ff2 - fi2)/timeLength*t*t
s2 = Ar * np.cos(phi)

# Noise --------------------------------------------------
noiseAmp = 0.8
noiseWidth = 0.4                       # seconds
noiseLocation = 0.8                     # seconds
noiseLocation2 = 1.8

# 0.3: shape parameter -> 0 per finestra rectangular
#                      -> 1 hanning
wnoise = get_window(("tukey", 0.3), int(noiseWidth*fs)) 

# two chirps
ss = (s1 + s2)
# delete gradually the area
ss[int(noiseLocation*fs):(int(noiseLocation*fs)+int(noiseWidth*fs))] *= (1.0 - wnoise)
ss[int(noiseLocation2*fs):(int(noiseLocation2*fs)+int(noiseWidth*fs))] *= (1.0 - wnoise)

# Rumore grande quanto il chirp
noise = np.zeros(phi.size) 
noiseInsert = np.random.normal(0, noiseAmp, int(noiseWidth*fs))

# butterworth 13rd order lowpass
b, a = butter(13, 0.1) 
noiseInsert = lfilter(b, a, noiseInsert)
noise[int(noiseLocation*fs):(int(noiseLocation*fs)+int(noiseWidth*fs))] = wnoise*noiseInsert
noise[int(noiseLocation2*fs):(int(noiseLocation2*fs)+int(noiseWidth*fs))] = wnoise*noiseInsert

zs = (ss + noise)*ww

# PLOT AND SAVE

plt.figure(1, figsize=(15, 3))
plt.plot(t, zs, 'b', lw=1.5)
plt.axis([0, 2,-1.0,1.0])
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')

#sd.play(zs, fs)

plt.tight_layout()
plt.savefig('sine-synthesis.png')
plt.show()

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
plt.title('frequency crossing')
plt.show()

#filename = 'sines-noise-cross.wav'
#MonoWriter(filename = filename, sampleRate = fs)((z).astype(single))

UF.wavwrite(zs, fs, "{0}s-chirps[{1}-{2};{3}-{4}];noises[{5},{6},{7},{8}].wav".format(timeLength, fi1, ff1, fi2, ff2, noiseAmp, noiseWidth, noiseLocation, noiseLocation2))