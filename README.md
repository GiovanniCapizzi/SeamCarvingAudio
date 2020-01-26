# Seam carving for sound applications

This repository collects the used code sources and materials to experimentally apply the **seam carving** technique on impact sounds, as part of my master's thesis work.

The structure is the following:

```bash
├── e2e.py
├── example
│   ├── bell.wav
│   ├── bell_follow.png
│   ├── bell_resynth.png
│   ├── bell_resynth.wav
│   ├── modified_bell.png
│   └── modified_bell.wav
├── fixed_frequencies.py
├── sounds
│   ├── 2chirps-1noise.py
│   ├── 2chirps-2noises.py
│   ├── 3chirps-2noises.py
│   ├── campana.wav
│   ├── fmod.py
│   ├── qsweep.py
│   └── qsweep_harmonics.py
├── tools.py
└── variable_frequencies.py
```



## Requirements

These files are using [SMS-tools](https://github.com/MTG/sms-tools) as the main analysis framework. Before starting any of these files you have to install this package and put this files in the workspace folder, or fix their relative paths.



## e2e.py

This file is the most pure application of the seam carving algorithm to the spectrogram. Using the function **popMode** a seam is extracted with the function **findVerticalSeam**, then point-to-point values are collected following the evolution of the detected component through time.

## fixed_frequencies.py

This implementation is an evolution of the previous file. Similar to the previous code, using the popMode function, parameters will be computed from an extracted seam. Moreover, stationary parameters are produced, such as the peak frequency, magnitude and phase. Using **Schroeder backward integration** and estimation of the decay time is computed. 

## variable_frequencies.py

Analog to the previous, this is a variant of the first **e2e.py** implementation, which gives point-to-point values, collected through time, combined with the **fixed_frequencies.py** estimation of decay.

## tools.py

A collection of several utility functions, used to plot and analyze sounds.

## sounds

A folder which collects the main sounds used in this work, and the code used to generates them.

## example

This folders contains the extraction and resynthesis of a bell sound, used as a manipulation example of the extracted parameters.