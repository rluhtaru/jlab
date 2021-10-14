# ----------------------------------------------------------------
# Load libraries
# ----------------------------------------------------------------
import numpy as np
import math
import sys
from gwpy.timeseries import TimeSeries

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import lmfit
from lmfit import Model, minimize, fit_report, Parameters
from scipy.linalg.decomp_svd import null_space
from scipy.optimize import curve_fit
from scipy import stats

import os

# ----------------------------------------------------------------
# Set parameters
# ----------------------------------------------------------------
data = {
    "tutorial150914H1": {"file": 'data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5', "time": 1126259462.422},
    "tutorial150914L1": {"file": 'data/L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5', "time": 1126259462.422},
    "GW170104H1": {"file": "data/H-H1_GWOSC_4KHZ_R1-1167559921-32.hdf5", "time": 1167559936.6},
    "GW170608": {"file": "data/H-H1_GWOSC_4KHZ_R1-1180920447-4096.hdf5", "time": 1180922494.5},
    "GW170608-32sH1": {"file": "data/H-H1_GWOSC_4KHZ_R1-1180922479-32.hdf5", "time": 1180922494.5},
    "GW170814L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1186741846-32.hdf5", "time": 1186741861.5},
    "GW170818L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1187058312-32.hdf5", "time": 1187058327.1},
    "GW170823L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1187529241-32.hdf5", "time": 1187529256.5},
    "GW190412L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1239082247-32.hdf5", "time": 1239082262.2},
    "GW190421H1": {"file": "data/H-H1_GWOSC_4KHZ_R1-1239917939-32.hdf5", "time": 1239917954.3},
    "GW190424L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1240164411-32.hdf5", "time": 1240164426.1},
    "GW190519L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1242315347-32.hdf5", "time": 1242315362.4},
    "GW190521L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1242442952-32.hdf5", "time": 1242442967.4},
    "GW190521_0743L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1242459842-32.hdf5", "time": 1242459857.5},
    "GW190602L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1243533570-32.hdf5", "time": 1243533585.1},
    "GW190620L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1245035064-32.hdf5", "time": 1245035079.3},
    "GW190630L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1245955928-32.hdf5", "time": 1245955943.2},
    "GW190708L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1246663500-32.hdf5", "time": 1246663515.4},
    "GW190814L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1249852241-32.hdf5", "time": 1249852257.0},
    "GW190828L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1251009248-32.hdf5", "time": 1251009263.8},
    "GW190828H1": {"file": "data/H-H1_GWOSC_4KHZ_R1-1251009248-32.hdf5", "time": 1251009263.8},
    "GW190910H1": {"file": "data/H-H1_GWOSC_4KHZ_R1-1251009248-32.hdf5", "time": 1251009263.8},
    "GW190910L1": {"file": "data/L-L1_GWOSC_4KHZ_R1-1252150090-32.hdf5", "time": 1252150105.3},
}
detector = 'H1'
evtname = "GW190828" + detector
fn = data[evtname]["file"]
tevent = data[evtname]["time"]

# ----------------------------------------------------------------
# Load LIGO data
# ----------------------------------------------------------------
strain = TimeSeries.read(fn, format='hdf5.losc')
center = int(tevent)
strain = strain.crop(center-16, center+16)
# ----------------------------------------------------------------
# Show LIGO strain vs. time
# ----------------------------------------------------------------
strainfig = strain.plot()
strainfig.gca().set_ylabel('strain')
# strainfig.show()

# ----------------------------------------------------------------
# Obtain the power spectrum density PSD / ASD
# ----------------------------------------------------------------

asd = strain.asd(fftlength=8)

asdfig = asd.plot()
asdax = asdfig.gca()
asdax.set_xlim(10, 2000)
asdax.set_ylim(1e-24, 1e-19)
asdax.set_ylabel('ASD (strain/Hz$^{1/2})$')
asdax.set_xlabel('Frequency (Hz)')
# asdfig.show()

# ----------------------------------------------------------------
# Whitening data
# ----------------------------------------------------------------

white_data = strain.whiten()

whitefig = white_data.plot()
whitefig.gca().set_ylabel('strain (whitened)')
# whitefig.show()

# ----------------------------------------------------------------
# Bandpass filtering
# ----------------------------------------------------------------

bandpass_low = 30
bandpass_high = 350

white_data_bp = white_data.bandpass(bandpass_low, bandpass_high)
max_signal_time = white_data_bp.times[np.argmax(white_data_bp.value)].value

bandpassfig = white_data_bp.plot()
ax = bandpassfig.gca()
ax.set_ylabel('strain (whitened + band-pass)')
ax.vlines(max_signal_time, *ax.get_ylim(), colors="red", lw=1)
ax.vlines(tevent, *ax.get_ylim(), colors="blue", lw=1)
bandpassfig.show()

# Zoom in the interesting region

zoomfig = white_data_bp.plot()
ax = zoomfig.gca()
ax.set_ylabel('strain (whitened + band-pass), zoomed in')
ax.set_xlim(tevent-0.17, tevent+0.13)
ax.set_ylim(-2, 2)
ax.vlines(tevent, *ax.get_ylim(), colors="red", lw=1)
zoomfig.show()

# ----------------------------------------------------------------
# q-transform
# ----------------------------------------------------------------

dt = 1  # -- Set width of q-transform plot, in seconds
hq = white_data_bp.q_transform(outseg=(tevent-dt, tevent+dt))
hq.shift(-hq.xindex[0])
print(hq.t0)
print(hq.times)
print(hq.xindex)

print(hq)


hqfig = hq.plot()
ax = hqfig.gca()
hqfig.colorbar(label="Normalised energy")
ax.grid(False)
# ax.set_yscale("log")
ax.set_xlim(dt - 0.5, dt + 0.5)
# ax.set_xlim(tevent-0.5, tevent+0.5)
ax.set_ylim(20, 400)
ax.set_ylabel('Frequency (Hz)')
# hqfig.show()

# ----------------------------------------------------------------
# Q-transform fit
# ----------------------------------------------------------------

energy_values = hq.value  # 2d array, x-axis is time, y-axis is freq, value is energy
max_freq_indices = np.argmax(energy_values, axis=1)
max_time_indices = np.argmax(energy_values, axis=0)

max_freqs = [hq.yindex[i].value for i in max_freq_indices]
max_times = [hq.xindex[i].value for i in max_time_indices]
times_in_seconds = [t.value for t in hq.xindex]
freqs_in_hertz = [f.value for f in hq.yindex]
# ax.scatter(times_in_seconds, max_freqs, c='red', marker='x')
# hqfig.show()

# Find global maximum energy, base_index is the x index of it
# We use unravel_index to reshape index because argmax flattens the array if no
# axis is given
base_index = np.unravel_index(np.argmax(energy_values), energy_values.shape)
base_time_index = base_index[0]
base_freq_index = base_index[1]
# ax.scatter(times_in_seconds[base_time_index], max_freqs[base_time_index], s=30, c='black', marker='s', zorder=10)

# # EQUAL TIME SEPARATION:

# # End iterating if maximum frequency differs more than CUTOFF_RATIO times, used
# # to determine where the interesting line starts and ends
# CUTOFF_RATIO = 1.2

# # x and y values of the data points on the interesting line
# line_times = [times_in_seconds[base_time_index]]
# line_freqs = [max_freqs[base_time_index]]

# # Check times after base time
# for i in range(base_time_index + 1, len(times_in_seconds)):
#     new_time = times_in_seconds[i]
#     new_freq = max_freqs[i]
#     ratio = new_freq / line_freqs[-1]
#     if (ratio > CUTOFF_RATIO or 1/ratio > CUTOFF_RATIO):
#         break

#     line_times.append(new_time)
#     line_freqs.append(new_freq)

# # Check times before base time
# for i in range(base_time_index - 1, 0, -1):
#     new_time = times_in_seconds[i]
#     new_freq = max_freqs[i]
#     ratio = new_freq / line_freqs[0]
#     if (ratio > CUTOFF_RATIO or 1/ratio > CUTOFF_RATIO):
#         break

#     line_times.insert(0, new_time)
#     line_freqs.insert(0, new_freq)

# EQUAL FREQUENCY SEPARATION:


def find_FWHM(max_time_index, freq_index):
    '''
    Find full width at half maximum in time-direction for a given time and freq
    index. The time index is assumed to be the index where the energy is maximal
    for this frequency.

    Returns (center time in seconds, FWMH in seconds)
    '''
    max_value = energy_values[max_time_index, freq_index]
    min_time = times_in_seconds[max_time_index]
    max_time = times_in_seconds[max_time_index]

    # Check larger times
    for i in range(max_time_index + 1, energy_values.shape[0]):
        value = energy_values[i, freq_index]
        if (value < max_value / 2):
            # Take the average of two times
            max_time = (max_time + times_in_seconds[i]) / 2
            break
        max_time = times_in_seconds[i]

    # Check smaller times
    for i in range(max_time_index - 1, 0, -1):
        value = energy_values[i, freq_index]
        if (value < max_value / 2):
            min_time = (min_time + times_in_seconds[i]) / 2
            break
        min_time = times_in_seconds[i]

    avg = (min_time + max_time) / 2
    return (avg, max_time - min_time)


def find_center(max_time_index, freq_index):
    return find_FWHM(max_time_index, freq_index)[0]


def find_time_uncertainty(max_time_index, freq_index, min_uncertainty=0):
    # FWHM is 2.355 sigmas
    uncertainty = find_FWHM(max_time_index, freq_index)[1] / 2.355
    return max(uncertainty, min_uncertainty)


# x and y values of the data points on the interesting line
line_times = [find_center(max_time_indices[base_freq_index], base_freq_index)]
line_time_uncertainties = [find_time_uncertainty(
    max_time_indices[base_freq_index], base_freq_index)]
line_freqs = [freqs_in_hertz[base_freq_index]]

# End iterating in each frequency direction if time of max value differs by more
# than TIME_CUTOFF
TIME_CUTOFF = 0.02

# Check times after base time
for i in range(base_freq_index + 1, len(freqs_in_hertz)):
    new_time = max_times[i]
    new_freq = freqs_in_hertz[i]
    diff = new_time - line_times[-1]
    if (abs(diff) > TIME_CUTOFF):
        break

    line_times.append(find_center(max_time_indices[i], i))
    line_time_uncertainties.append(
        find_time_uncertainty(max_time_indices[i], i))
    line_freqs.append(new_freq)

# Check times before base time
for i in range(base_freq_index - 1, 0, -1):
    new_time = max_times[i]
    new_freq = freqs_in_hertz[i]
    diff = new_time - line_times[0]
    if (abs(diff) > TIME_CUTOFF):
        break

    line_times.insert(0, find_center(max_time_indices[i], i))
    line_time_uncertainties.insert(
        0, find_time_uncertainty(max_time_indices[i], i))
    line_freqs.insert(0, new_freq)

print(line_times)
print(line_time_uncertainties)
print(line_freqs)
# Plot only the interesting line
# line_times = line_times[math.floor(len(line_times)/2):]
# line_freqs = line_freqs[math.floor(len(line_freqs)/2):]
# ax.scatter(line_times, line_freqs, c='xkcd:blue', marker='x')
PLOT_EVERY = 10
ax.errorbar(line_times[::PLOT_EVERY], line_freqs[::PLOT_EVERY],
            xerr=line_time_uncertainties[::PLOT_EVERY], c='xkcd:blue', marker='x', fmt="none")
# hqfig.show()

# Frequency model


def gwfreq(t, t0, M):
    '''
    Frequency (not angular frequency!) model for gravitational waves.
    t - time in seconds
    t0 - event time in seconds
    M - chirp mass in sun masses
    Returns f in Hz.
    '''
    const = 1/(2*np.pi) * 948.5 * np.power(1/M, 5/8)
    # set a max cutoff but not too large to affect fitting
    TIME_CUTOFF = 1e-5
    return const*np.power(np.maximum(t0-t, TIME_CUTOFF), -3/8)


def inv_gwfreq(f, t0, M):
    '''
    Inverse of gwfreq.
    '''
    return t0 - (948.5 / (2*np.pi))**(8/3) * np.power(1/M, 5/3) * np.power(f, -8/3)


def loggwfreq(t, t0, M): return np.log(gwfreq(t, t0, M))


# Fit log(freq)
# popt, pcov = curve_fit(loggwfreq, line_times, np.log(line_freqs), p0=[
#                        line_times[-1] + 0.1, 20], bounds=([line_times[-1], 1], [line_times[-1] + 0.3, 100]), verbose=1, xtol=None, maxfev=1e4, method='dogbox')

# Fit inverse
popt, pcov = curve_fit(inv_gwfreq, line_freqs, line_times, p0=[
                       line_times[-1], 20], sigma=line_time_uncertainties, absolute_sigma=True, bounds=([line_times[-1] - 0.3, 1], [line_times[-1] + 0.3, 100]), verbose=1, xtol=None, maxfev=1e4, method='dogbox')

print("Optimal parameters", popt)
print("Optimal parameter errors", np.sqrt(np.diag(pcov)))
chisq = np.sum(((inv_gwfreq(line_freqs, *popt) - line_times) /
               line_time_uncertainties)**2)
ddof = 2
reduced_chisq = chisq / (len(line_freqs) - ddof)
chiprob = 1 - stats.chi2.cdf(chisq, len(line_freqs) - ddof)
print("Chi^2", chisq)
print("Reduced chi^2", reduced_chisq)
print("Probability of chi^2", chiprob)
print("1 - Probability of chi^2", stats.chi2.cdf(chisq, len(line_freqs) - ddof))


ax.plot(times_in_seconds, gwfreq(times_in_seconds, *popt), c='red', zorder=10)
hqfig.show()
sys.exit()

# ----------------------------------------------------------------
# Frequency analytic
# ----------------------------------------------------------------


# def gwfreq(iM, iT, iT0):
#     const = (948.5)*np.power((1./iM), 5./8.)
#     # we can max it out above 500 Hz-ish
#     output = const*np.power(np.maximum((iT0-iT), 1e-2), -3./8.)
#     return output


times = np.linspace(0, 5, 50)
freq = gwfreq(times, 4, 20)

freqfig, ax = plt.subplots()
ax.plot(times, freq)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_yscale("log")
freqfig.show()

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
plt.show()

# define osc_dif for lmfit::minimize()


def osc_dif(params, x, data, eps):
    iM = params["Mc"]
    iT0 = params["t0"]
    norm = params["C"]
    phi = params["phi"]
    val = osc(x, iM, iT0, norm, phi)
    return (val-data)/eps

# ----------------------------------------------------------------
# Fit
# ----------------------------------------------------------------


sample_times = white_data_bp.times.value
sample_data = white_data_bp.value
low_cutoff = -0.05  # fit start, in seconds relative to highest signal
high_cutoff = 0.12  # fit end, in seconds relative to highest signal
indxt = np.where((sample_times >= (tevent + low_cutoff)) &
                 (sample_times < (tevent + high_cutoff)))
x = sample_times[indxt]
x = x-x[0]
white_data_bp_zoom = sample_data[indxt]

plt.figure(figsize=(12, 4))
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
plt.plot(x, white_data_bp_zoom)
plt.xlabel('Time (s)')
plt.ylabel('strain (whitened + band-pass)')

popt, pcov = curve_fit(osc, x, white_data_bp_zoom, [20, -low_cutoff, 1, 0])

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
plt.plot(x, osc(x, *popt), 'r', label='best fit')
plt.show()
