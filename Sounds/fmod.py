import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import get_window
sys.path.append("../../../../software/models")
import utilFunctions as UF
import stft as STFT



timeLength = 5                                 # seconds
fs = 44100
t = np.arange(0, timeLength, 1.0/fs)
ww = get_window("hamming", t.size)

modulator_frequency = 1.0
carrier_frequency = 440.0
                                                  
modulation_index = 10 / modulator_frequency    # frequency deviation => maximum_frequency_shift / modulator_frequency


# Am = -t* np.cos(2*np.pi*-t)#-t * np.cos(2*np.pi*t) #-t/modulator_frequency # np.log(np.e**-t) = -t
Am = -t/modulator_frequency                              # 1 or 2*np.pi or np.e**-t
Ac = np.e**-t
modulator = Am * np.sin(2.0 * np.pi * modulator_frequency * t) * modulation_index
s0 = Ac * np.cos(2.0 * np.pi * carrier_frequency * t + modulator)

Ac = np.e**-t/2
carrier_frequency = 650.0
s1 = Ac * np.cos(2.0 * np.pi * carrier_frequency * t + modulator)

Ac = np.e**-t/4
carrier_frequency = 860.0
s2 = Ac * np.cos(2.0 * np.pi * carrier_frequency * t + modulator)

Ac = np.e**-t/8
carrier_frequency = 1070.0
s3 = Ac * np.cos(2.0 * np.pi * carrier_frequency * t + modulator)

# Nessuna finestra applicata all'output. 
# Sinusoide costante, rumore filtrato "dinamicamente"
# https://en.wikipedia.org/wiki/Convex_combination
ss = (s0+s1+s2+s3)/4 
#ss = ss/np.max(ss)
zs = ss #* ww

# PLOT AND SAVE

plt.figure(1, figsize=(15, 3))
plt.plot(t, zs, 'b', lw=1.5)
plt.axis([0, timeLength,-1.0,1.0])
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')

plt.tight_layout()
plt.savefig('synthesis_modulation.png')
plt.show()

# STFT Analysis
M = 2*2048
Hop = 10*M//100
N = 2*M
maxplotfreq = 1400.0
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
plt.savefig('spectrogram_modulation.png')
plt.show()

# Writing File
UF.wavwrite(zs, fs, "output.wav")