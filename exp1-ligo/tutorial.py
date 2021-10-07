#----------------------------------------------------------------
# Load libraries
#----------------------------------------------------------------
import numpy as np
import math
from gwpy.timeseries import TimeSeries

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import lmfit
from lmfit import Model, minimize, fit_report, Parameters
from scipy.optimize import curve_fit

import os

#----------------------------------------------------------------
# Set parameters
#----------------------------------------------------------------
data = {
  "tutorial150914": 'data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5',
  "GW170104": "data/H-H1_GWOSC_4KHZ_R1-1167557889-4096.hdf5",
  "GW170608": "data/H-H1_GWOSC_4KHZ_R1-1180920447-4096.hdf5",
}
evtname = "GW170104"
fn = datas[evtname]
tevent = 1126259462.422 # Mon Sep 14 09:50:45 GMT 2015
evtname = 'GW150914' # event name

detector = 'H1' # detecotr: L1 or H1

#----------------------------------------------------------------
# Load LIGO data
#----------------------------------------------------------------
strain = TimeSeries.read(fn, format='hdf5.losc')
center = int(tevent)
strain = strain.crop(center-16, center+16)

#----------------------------------------------------------------
# Show LIGO strain vs. time
#----------------------------------------------------------------
strain.plot()
plt.ylabel('strain')
# plt.show()

#----------------------------------------------------------------
# Obtain the power spectrum density PSD / ASD
#----------------------------------------------------------------

asd = strain.asd(fftlength=8)

plt.close()
asd.plot()
plt.xlim(10, 2000)
plt.ylim(1e-24, 1e-19)
plt.ylabel('ASD (strain/Hz$^{1/2})$')
plt.xlabel('Frequency (Hz)')
# plt.show()

#----------------------------------------------------------------
# Whitening data
#----------------------------------------------------------------

white_data = strain.whiten()

plt.close()
white_data.plot()
plt.ylabel('strain (whitened)')
# plt.show()

#----------------------------------------------------------------
# Bandpass filtering
#----------------------------------------------------------------

bandpass_low = 30
bandpass_high = 350

white_data_bp = white_data.bandpass(bandpass_low, bandpass_high)

plt.close()
white_data_bp.plot()
plt.ylabel('strain (whitened + band-pass)')
# plt.show()

# Zoom in the interesting region

max_signal_time = white_data_bp.times[np.argmax(white_data_bp.value)].value

plt.close()
white_data_bp.plot()
plt.ylabel('strain (whitened + band-pass), zoomed in')
plt.xlim(max_signal_time-0.17, max_signal_time+0.13)
print(plt.gca().get_xlim())
plt.vlines(max_signal_time, *plt.gca().get_ylim(), colors="blue", lw=1)
plt.vlines(tevent, *plt.gca().get_ylim(), colors="red", lw=1)
# plt.show()

#----------------------------------------------------------------
# q-transform
#----------------------------------------------------------------

dt = 1  #-- Set width of q-transform plot, in seconds
hq = strain.q_transform(outseg=(tevent-dt, tevent+dt))

plt.close()
fig = hq.plot()
ax = fig.gca()
fig.colorbar(label="Normalised energy")
ax.grid(False)
plt.xlim(tevent-0.5, tevent+0.4)
plt.ylim(0, 400)
plt.ylabel('Frequency (Hz)')
# plt.show()

#----------------------------------------------------------------
# Frequency analytic
#----------------------------------------------------------------

def gwfreq(iM,iT,iT0):
    const = (948.5)*np.power((1./iM),5./8.)
    output = const*np.power(np.maximum((iT0-iT),1e-2),-3./8.) # we can max it out above 500 Hz-ish
    return output

times = np.linspace(0., 4., 50)
freq = gwfreq(20, times, 4)

plt.clf()
plt.plot(times, freq)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
# plt.show()

# Model
def osc(t, Mc, t0, C, phi):
  freq = gwfreq(Mc, t, t0)
  deltaT = t0 - t
  damping = np.power(1e-30, np.heaviside(t - t0, 0) * (t - t0))
  return C * np.power(Mc * freq, 10/3) * np.cos(freq * deltaT + phi) * damping

# Draw the function defined
times = np.linspace(-0.1, 0.2, 1000)
freq = osc(times, 30, 0.18, 1, 0.0)
plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(times, freq)
plt.xlabel('Time (s) since '+str(tevent))
plt.ylabel('strain')
# plt.show()

# define osc_dif for lmfit::minimize()
def osc_dif(params, x, data, eps):
    iM=params["Mc"]
    iT0=params["t0"]
    norm=params["C"]
    phi=params["phi"]
    val=osc(x, iM, iT0, norm, phi)
    return (val-data)/eps

#----------------------------------------------------------------
# Fit
#----------------------------------------------------------------

sample_times = white_data_bp.times.value
sample_data = white_data_bp.value
low_cutoff = -0.10 # fit start, in seconds relative to highest signal
high_cutoff = 0 # fit end, in seconds relative to highest signal
indxt = np.where((sample_times >= (max_signal_time + low_cutoff)) & (sample_times < (max_signal_time + high_cutoff)))
x = sample_times[indxt]
x = x-x[0]
white_data_bp_zoom = sample_data[indxt]

plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(x, white_data_bp_zoom)
plt.xlabel('Time (s)')
plt.ylabel('strain (whitened + band-pass)')

popt, pcov = curve_fit(osc, x, white_data_bp_zoom, [17, 0.10, 1, 0])

print("Optimal parameters:", popt)

# model = lmfit.Model(osc)
# p = model.make_params()
# p['Mc'].set(17.1)     # Mass guess
# p['t0'].set(0.17)  # By construction we put the merger in the center
# p['C'].set(1)      # normalization guess
# p['phi'].set(0)    # Phase guess
# unc = np.full(len(white_data_bp_zoom),20)
# out = minimize(osc_dif, params=p, args=(x, white_data_bp_zoom, unc))
# print(fit_report(out))
plt.plot(x, osc(x, *popt),'r',label='best fit')
plt.show()

