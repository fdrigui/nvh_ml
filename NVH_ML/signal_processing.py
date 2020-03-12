# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:15:39 2020

@author: RiguiFD
"""

import numpy as np
from scipy.signal import welch, spectrogram, butter, lfilter, find_peaks
import pandas as pd
from scipy import stats, signal
from scipy.fftpack import fft

# Plotting
import matplotlib.pyplot as plt
import matplotlib.style

# %matplotlib inline
matplotlib.style.use('ggplot')


def stats_summ(sgnl, time, sgnl_param, param, name):

    ftrd_sgnl = butter_bandpass_filter(sgnl,
                                       int(param['cutting_freq']['f_min']),
                                       int(param['cutting_freq']['f_max']),
                                       sgnl_param['fs'])
    std = np.std(ftrd_sgnl)
    var = np.var(ftrd_sgnl)
    abs_sgnl = np.absolute(ftrd_sgnl)
    mv_avg_abs_data = sma(abs_sgnl, int(float(sgnl_param['fs'])/10))
    mean = np.mean(abs_sgnl)
    maximum = np.amax(abs_sgnl)
    minimum = np.amin(abs_sgnl)
    x_axis_mv_avg_abs_data = np.arange(0, mv_avg_abs_data.size, 1)
    linreg = stats.linregress(x_axis_mv_avg_abs_data, mv_avg_abs_data)
    polyn, res, _, _, _ = np.polyfit(x_axis_mv_avg_abs_data, mv_avg_abs_data,
                                     3, full=True)
    qtl_20 = np.quantile(abs_sgnl, 0.2)
    qtl_50 = np.quantile(abs_sgnl, 0.5)
    qtl_80 = np.quantile(abs_sgnl, 0.8)
    summary = {'{}_std'.format(name): [std],
               '{}_var'.format(name): [var],
               '{}_mean'.format(name): [mean],
               '{}_max'.format(name): [maximum],
               '{}_min'.format(name): [minimum],
               '{}_lreg_slope'.format(name): [linreg.slope],
               '{}_lreg_intercept'.format(name): [linreg.intercept],
               '{}_lieg_rvalue'.format(name): [linreg.rvalue],
               '{}_lreg_pvalue'.format(name): [linreg.pvalue],
               '{}_lreg_stderr'.format(name): [linreg.stderr],
               '{}_polyreg_3'.format(name): [polyn[0]],
               '{}_polyreg_2'.format(name): [polyn[1]],
               '{}_polyreg_1'.format(name): [polyn[2]],
               '{}_polyreg_0'.format(name): [polyn[3]],
               '{}_polyreg_resid'.format(name): [res[0]],
               '{}_qtl_20'.format(name): [qtl_20],
               '{}_qtl_50'.format(name): [qtl_50],
               '{}_qtl_80'.format(name): [qtl_80]}
    return pd.DataFrame.from_dict(summary)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# F I L T E R I N G
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def butter_bandpass(lowcut, upcut, fs, order=2):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = upcut / nyq
    b, a = butter(order, [low, high], 'bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, upcut, fs, order=2):

    b, a = butter_bandpass(lowcut, upcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def sma(data: np.ndarray, n: int) -> np.ndarray:

    weights = np.ones(n)/n
    return np.convolve(weights, data)[n:-n]


def smoothing_to_get_peaks(sgnl, x_sgnl, param):
    if param['do']:
        window = signal.general_gaussian(param['M'],
                                         p=param['p'],
                                         sig=param['sig'])
        ftrd_sgnl = signal.fftconvolve(window, sgnl)
        ftrd_sgnl = (np.average(sgnl) / np.average(ftrd_sgnl)) * ftrd_sgnl
        ftrd_sgnl = ftrd_sgnl[:((-1)*param['M'])+1]
    else:
        ftrd_sgnl = sgnl

    if param['plot']:
        plt.plot(x_sgnl, ftrd_sgnl)
        plt.xlabel('Frequency [Hz]', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.title("Gausian filtering of Freq. Domain FFT", fontsize=12)
        plt.show()

    return ftrd_sgnl

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# F A S T   F O U R R I E R   T R A N S F O R M
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_fft_values(sgnl, sgnl_param, param):

    fd = 1.0 / (2.0 * sgnl_param['dt']) / float(sgnl_param['N'] // 2)
    fft_ini = int(param['f_min'] / fd)
    fft_fin = int(param['f_max'] / fd)
    f_val_size = int((param['f_max'] - param['f_min']) / fd)
    fft_f_val = np.linspace(int(param['f_min']), int(param['f_max']),
                            f_val_size)
    fft_val = fft(sgnl)
    fft_val = 2.0 / sgnl_param['N'] * np.abs(fft_val[fft_ini:fft_fin])
    fft_fd = fd

    if param['plot']:
        plt.plot(fft_f_val[fft_ini:fft_fin], fft_val[fft_ini:fft_fin])
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.show()

    return fft_f_val[fft_ini:fft_fin], fft_val[fft_ini:fft_fin], fft_fd


def get_fft_peaks(sgnl, sgnl_param, param):
    list_10_val, list_10_x_val = [], []
    for p in param:
        fft_f_val, fft_val, fft_fd = get_fft_values(sgnl, sgnl_param, p)
        ftrd_sgnl = smoothing_to_get_peaks(fft_val, fft_f_val, p['smoothing'])
        peaks = get_peaks(ftrd_sgnl, fft_f_val, int(300/fft_fd), int(21/fft_fd),
                          p['peaks'])
        _10_val, _10_x_val = get_top_n(ftrd_sgnl, peaks, fft_fd,
                                       p['f_min'], p['get_top_10'])
        list_10_val.append(_10_val)
        list_10_x_val.append(_10_x_val)
    return _10_val, _10_x_val


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# P O W E R   S P E C T R U M   D E N S I T Y
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_psd_values(sgnl, sgnl_param, param):

    psd_f_val, psd_val = welch(sgnl, sgnl_param['fs'],
                               window=param['welch']['window'],
                               nperseg=param['welch']['blocksize'],
                               noverlap=param['welch']['overlap']*param['welch']['blocksize'],
                               return_onesided=True, scaling=param['welch']['scaling'],
                               average=param['welch']['average'])

    fd = psd_f_val[-1]/psd_f_val.size
    psd_ini = int(param['cutting_freq']['f_min'] / fd)
    psd_fin = int(param['cutting_freq']['f_max'] / fd)

    if param['plot']:
        plt.plot(psd_f_val[psd_ini:psd_fin], psd_val[psd_ini:psd_fin])
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.show()

    return psd_f_val[psd_ini:psd_fin], psd_val[psd_ini:psd_fin], fd


def get_psd_peaks(sgnl, sgnl_param, param, t_name):

    psd_f_val, psd_val, psd_fd = get_psd_values(sgnl, sgnl_param, param)
    ftrd_sgnl = smoothing_to_get_peaks(psd_val, psd_f_val, param['smoothing'])
    peaks = get_peaks(ftrd_sgnl, psd_f_val, int(300/psd_fd), int(21/psd_fd),
                      param['peaks'])
    psd_peaks = get_top_n(ftrd_sgnl, peaks, psd_fd,
                          param['cutting_freq']['f_min'],
                          param['get_top_n'], t_name)

    return psd_peaks

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A U T O   C O R R E L A T I O N
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_autocorr_values(sgnl, sgnl_param, param):
    acorr_val = np.correlate(sgnl, sgnl, mode='full')
    acorr_val = acorr_val[(len(acorr_val)//2)+100:]
    acorr_x_val = []
    for jj in range(0, sgnl_param['N']-100):
        acorr_x_val.append(sgnl_param['dt'] * jj)

    acorr_x_val = np.asarray(acorr_x_val, np.ndarray)
    plt.plot(acorr_x_val, acorr_val)
    plt.xlabel('time [s]', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.title("autocorrelation of sgnal", fontsize=16)
    plt.show()
    return acorr_x_val, acorr_val


def get_acorr_peaks(sgnl, sgnl_param, param):
    list_10_val, list_10_x_val = [], []
    for p in param:
        acorr_x_val, acorr_val = get_autocorr_values(sgnl, sgnl_param, p['t_min'])
        acorr_val = np.abs(acorr_val)
        ftrd_sgnl = smoothing_to_get_peaks(acorr_val, acorr_x_val, p['smoothing'])
        peaks = get_peaks(ftrd_sgnl, acorr_x_val, 6000, 600,
                          p['peaks'])
        _10_val, _10_x_val = get_top_n(ftrd_sgnl, peaks, sgnl_param['dt'],
                                       0, p['get_top_10'])
        return list_10_val, list_10_x_val


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# W A V E L E T   T R A N S F O R M A T I O N
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# F I N D I N G   P E A K S
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_peaks(sgnl, x_sgnl, distance, width, param):
    peaks, test = find_peaks(sgnl, distance=distance, width=width)
    if param['plot']:
        plt.plot(x_sgnl, sgnl)
        plt.plot(x_sgnl[peaks], sgnl[peaks], "x")
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Peak selection on FFT", fontsize=16)
        plt.show()
    return peaks


# def get_top_n(sgnl, peaks, xd, fmin, param, t_name):
#     n = param['n']
#     peaks_val_list = sgnl[peaks]
#     n_val = np.sort(peaks_val_list)[-n:][::-1]
#     n_x_val = []
#     y = pd.DataFrame()
#     i = 0
#     for peak in n_val:
#         n_x_val.append((peaks[peaks_val_list == peak][0]*xd)+fmin)
#         y = y.join(pd.DataFrame.from_dict({'{}_Amp{}'.format(t_name, i): peak,
#                                            '{}_x{}'.format(t_name, int(i)): (peaks[peaks_val_list == peak][0]*xd)+fmin}))
#         i += 1
#     n_x_val = np.asarray(n_x_val, np.ndarray)
#     if n_val.size < n:
#         fill_zeros = np.zeros(n - n_val.size)
#         n_val = np.append(n_val, fill_zeros)
#         n_x_val = np.append(n_x_val, fill_zeros)
#     if param['print']:
#         values = zip(n_val, n_x_val)

#         for val, x_val in values:
#             print('Freq:{0:1.2f}Hz;\tAmp:{0:1.2e}'.format(x_val, val))
#     print(n_val)
#     return n_val, n_x_val, y

def get_top_n(sgnl, peaks, xd, fmin, param, t_name):
    n = param['n']
    peaks_val_list = sgnl[peaks]
    n_val = np.sort(peaks_val_list)[-n:][::-1]

    if n_val.size < n:
        n_val = np.append(n_val, np.zeros(n - n_val.size))
    result = {}
    i = 0
    for val in n_val:
        result['{}_Amp_{}'.format(t_name, i)] = [val]
        result['{}_x_{}'.format(t_name, i)] = [(peaks[peaks_val_list == val][0]*xd)+fmin]
        i += 1

    result_df = pd.DataFrame.from_dict(result)
    return result_df