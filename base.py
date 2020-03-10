# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:12:03 2020

@author: RiguiFD

Reference: http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
"""

# Mathematics
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import stats, signal
from scipy.signal import welch, spectrogram, butter, lfilter, find_peaks
import pandas as pd
from detect_peaks import detect_peaks

# File Handler
import nptdms
# from tkinter import *
# from tkinter.filedialog import askopenfilename
import os

# Plotting
import matplotlib.pyplot as plt
import matplotlib.style

# %matplotlib inline
matplotlib.style.use('ggplot')

__author__ = "Filipi Rigui, https://github.com/fdrigui/NVH_Analysis"
__version__ = "0.0.1"
__license__ = "MIT"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# F i l e   M a n a g i n g
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# def choose_file() -> str:
#     '''
#     Return a path string of a desired file.

#     Returns
#     -------
#     str
#         path of the desired file.

#     '''
#     root = Tk()
#     name = askopenfilename(initialdir='C:/',
#                            filetypes=(("TDMS", "*.tdms"), ("all", "*.*")),
#                            title='Choose an TDMS result')
#     print("The file that the user has choosen was: '"+name+"'.")
#     try:
#         with open(name, 'r'):
#             file_path = name
#             print('File Found')
#     except FileNotFoundError:
#         file_path = ""
#         print('file not found')
#     finally:
#         root.destroy()
#     return file_path


def list_files(folder_path, file_extension):
    '''
    Description of Function

    Parameters
    ----------
    folder_path : TYPE
        DESCRIPTION.
    file_extension : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    files = [fn for fn in os.listdir(folder_path)
             if fn.endswith(file_extension)]
    return files


def load_signal_from_tdms(file_path: str, group_name: str,
                          channel_name: str) -> np.ndarray:
    '''
    Open a TDMS file and extract a single signal from that.
    Parameters
    ----------
    str : file_path
        The Path of TDMS file that the user wants to extract the signal.
    str : group_name
        Name of group that contains the signal of interest.
    str : channel_name
        Name of signal of interest.

    Returns
    -------
    data : np.ndarray
        The signal itself.
    time : np.ndarray
        The time series of the signal.

    '''
    tdms_file = nptdms.TdmsFile(file_path)

    channel = tdms_file.object(group_name, channel_name)

    data = channel.data
    time = channel.time_track()

    file_name = file_path[::-1][:file_path[::-1].find('/')][::-1]
    t_n = time[-1]
    N = data.size
    dt = t_n/N
    fs = int(round(np.reciprocal(dt), 0))
    dt = np.reciprocal(float(fs))
    data_param = {'t_n': t_n, 'N': N, 'dt': dt, 'fs': fs, 'f_name': file_name}

    return data, time, data_param

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# F i l t e r i n g   a n d   S l i c i n g
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def butter_bandpass(lowcut: int, upcut: int, fs: int,
                    order: int = 2) -> np.ndarray:
    '''
    Butterworth Digital Filter.
    Design an Nth order digital Butterworth filter and return the filter
    coefficients in (B,A)

    Parameters
    ----------
    lowcut : int
        Lower threshold for frequencies. All freqs lower than that are
        filtered.
    upcut : int
        Higher threshold for frequencies. All freqs lower than that are
        filtered.
    fs : int
        Frequency Sample Rate, is the frequency the data get sampled
        (acquired).
    order : int, optional
        The order of the filter. The default is 2.

    Returns
    -------
    b : TYPE
        The numerator of IRR Filter.
    a : TYPE
        The denominator of IRR Filter.

    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = upcut / nyq
    b, a = butter(order, [low, high], 'bandpass')
    return b, a


def butter_bandpass_filter(data: np.ndarray, lowcut: int, upcut: int, fs: int,
                           order: int = 2):
    '''
    Digital filter using Butterworth coeficients on lfilter.

    Parameters
    ----------
    data : np.ndarray
        Input array.
    lowcut : int
        Lower threshold for frequencies. All freqs lower than that are
        filtered.
    upcut : int
        Higher threshold for frequencies. All freqs lower than that are
        filtered.
    fs : int
        Frequency Sample Rate, is the frequency the data get sampled
        acquired).
    order : int, optional
        The order of the filter. The default is 2.

    Returns
    -------
    y : array
        The output of the digital filter.

    '''
    b, a = butter_bandpass(lowcut, upcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def sma(data: np.ndarray, n: int) -> np.ndarray:
    '''
    Simple Moving Average. Uses Convolution method to calculate the mean based
    on n window size

    Parameters
    ----------
    data : np.ndarray
        Array like input.
    n : TYPE
        Window size.

    Returns
    -------
    np.ndarray
        averaged array.

    '''
    weights = np.ones(n)/n
    return np.convolve(weights, data)[n:-n]


def slicing_by_time(sgnl, sgnl_param, param):
    '''
    Slice a array based on time in seconds

    Parameters
    ----------
    data : np.ndarray
        Array that wants to be sliced.
    param : array
        array like, where 1st element is the start time and 2nd is end time in
        seconds. ex: [2.0, 5.0] to slice from 2s to 5s
    fs : frequency sampled
        it is the acquisition rate the data where acquired (sampled).

    Returns
    -------
    np.ndarray
        portion of data within parameters.

    '''
    start = param[0]
    end = param[1]
    sliced_sgnl = sgnl[int(start*sgnl_param['fs']):int(end*sgnl_param['fs'])]

    sgnl_param['N'] = sliced_sgnl.size
    sgnl_param['t_n'] = sgnl.size*sgnl_param['dt']
    sgnl_param['dt'] = sgnl_param['dt']
    sgnl_param['fs'] = sgnl_param['fs']
    sgnl_param['f_name'] = sgnl_param['f_name']

    return sliced_sgnl, sgnl_param

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# S i g n a l   p r o c e s s i n g
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def statistics_summary(data: np.ndarray, name, lowcut, upcut, fs):
    '''
    returns statistical summary based on input data

    Parameters
    ----------
    data : np.ndarray
        DESCRIPTION.
    lowcut : TYPE
        DESCRIPTION.
    upcut : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.

    Returns
    -------
    summary : TYPE
        DESCRIPTION.

    '''
    data = butter_bandpass_filter(data, lowcut, upcut, fs)
    std = np.std(data)
    var = np.var(data)
    abs_data = np.absolute(data)
    mv_avg_abs_data = sma(abs_data, int(float(fs)/10))
    mean = np.mean(abs_data)
    maximum = np.amax(abs_data)
    minimum = np.amin(abs_data)
    x_axis_mv_avg_abs_data = np.arange(0, mv_avg_abs_data.size, 1)
    linreg = stats.linregress(x_axis_mv_avg_abs_data, mv_avg_abs_data)
    polyn, res, _, _, _ = np.polyfit(x_axis_mv_avg_abs_data, mv_avg_abs_data,
                                     3, full=True)
    qtl_20 = np.quantile(abs_data, 0.2)
    qtl_50 = np.quantile(abs_data, 0.5)
    qtl_80 = np.quantile(abs_data, 0.8)
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


def get_fft_values(sgnl, sgnl_param, param):
    '''
    Description

    Parameters
    ----------
    sgnl : TYPE
        DESCRIPTION.
    sgnl_param : TYPE
        DESCRIPTION.

    Returns
    -------
    f_val : TYPE
        DESCRIPTION.
    fft_val : TYPE
        DESCRIPTION.

    '''

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


def get_psd_values(sgnl, sgnl_param, param):
    '''
    Description

    Parameters
    ----------
    sgnl : TYPE
        DESCRIPTION.
    sgnl_param : TYPE
        DESCRIPTION.
    param : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    psd_f_val, psd_val = welch(sgnl, sgnl_param['fs'],
                               window=param['welch']['window'],
                               nperseg=param['welch']['blocksize'],
                               noverlap=param['welch']['overlap']*param['welch']['blocksize'],
                               return_onesided=True, scaling=param['welch']['scaling'],
                               average=param['welch']['average'])

    fd = psd_f_val[-1]/psd_f_val.size
    psd_ini = int(param['f_min'] / fd)
    psd_fin = int(param['f_max'] / fd)

    if param['plot']:
        plt.plot(psd_f_val[psd_ini:psd_fin], psd_val[psd_ini:psd_fin])
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.show()

    return psd_f_val[psd_ini:psd_fin], psd_val[psd_ini:psd_fin], fd


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
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Gausian filtering of Freq. Domain FFT", fontsize=16)
        plt.show()

    return ftrd_sgnl


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


def get_top_10(sgnl, peaks, xd, fmin, param):
    peaks_val_list = sgnl[peaks]
    _10_val = np.sort(peaks_val_list)[-10:][::-1]
    _10_x_val = []
    for peak in _10_val:
        _10_x_val.append((peaks[peaks_val_list == peak][0]*xd)+fmin)
    _10_x_val = np.asarray(_10_x_val, np.ndarray)
    if _10_val.size < 10:
        fill_zeros = np.zeros(10 - _10_val.size)
        _10_val = np.append(_10_val, fill_zeros)
        _10_x_val = np.append(_10_x_val, fill_zeros)
    if param['print']:
        values = zip(_10_val, _10_x_val)

        for val, x_val in values:
            print('Freq:{0:1.2f}Hz;\tAmp:{0:1.2e}'.format(x_val, val))
    print(_10_val)
    return _10_val, _10_x_val


def get_fft_peaks(sgnl, sgnl_param, param):
    list_10_val, list_10_x_val = [], []
    for p in param:
        fft_f_val, fft_val, fft_fd = get_fft_values(sgnl, sgnl_param, p)
        ftrd_sgnl = smoothing_to_get_peaks(fft_val, fft_f_val, p['smoothing'])
        peaks = get_peaks(ftrd_sgnl, fft_f_val, int(300/fft_fd), int(21/fft_fd),
                          p['peaks'])
        _10_val, _10_x_val = get_top_10(ftrd_sgnl, peaks, fft_fd,
                                        p['f_min'], p['get_top_10'])
        list_10_val.append(_10_val)
        list_10_x_val.append(_10_x_val)
    return _10_val, _10_x_val


def get_psd_peaks(sgnl, sgnl_param, param):
    list_10_val, list_10_x_val = [], []
    for p in param:
        psd_f_val, psd_val, psd_fd = get_psd_values(sgnl, sgnl_param, p)
        ftrd_sgnl = smoothing_to_get_peaks(psd_val, psd_f_val, p['smoothing'])
        peaks = get_peaks(ftrd_sgnl, psd_f_val, int(300/psd_fd), int(21/psd_fd),
                          p['peaks'])
        _10_val, _10_x_val = get_top_10(ftrd_sgnl, peaks, psd_fd,
                                        p['f_min'], p['get_top_10'])
        list_10_val.append(_10_val)
        list_10_x_val.append(_10_x_val)
    return list_10_val, list_10_x_val


def get_acorr_peaks(sgnl, sgnl_param, param):
    list_10_val, list_10_x_val = [], []
    for p in param:
        acorr_x_val, acorr_val = get_autocorr_values(sgnl, sgnl_param, p['t_min'])
        acorr_val = np.abs(acorr_val)
        ftrd_sgnl = smoothing_to_get_peaks(acorr_val, acorr_x_val, p['smoothing'])
        peaks = get_peaks(ftrd_sgnl, acorr_x_val, 6000, 600,
                          p['peaks'])
        _10_val, _10_x_val = get_top_10(ftrd_sgnl, peaks, sgnl_param['dt'],
                                        0, p['get_top_10'])
        return list_10_val, list_10_x_val

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A n a l y s i s
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def analysis(sgnl, sgnl_param, param):
    df = pd.DataFrame({'file_name': [sgnl_param['f_name']]})
    for i in param['analysisTime']:
        # Slices the signal on each segment of interest
        slcd_sgnl, slcd_sgnl_param = slicing_by_time(sgnl, [i['start'],
                                                            i['end']], 25600)
        # Apply statistical analysis over the signal and merge on df DataFrame
        stat = statistics_summary(slcd_sgnl, i['name'], 300, 6000, 25600)
        df = df.join(stat)

    return df


def full_analysis(full_param):
    files = list_files(full_param['folder_path'], full_param['file_extension'])

    result = pd.DataFrame()
    for file in files:
        file_path = full_param['folder_path'] + file
        sgnl, _, sgnl_par = load_signal_from_tdms(file_path,
                                                  full_param['group_name'],
                                                  full_param['channel_name'])
        x = analysis(sgnl, full_param, file)
        result = result.append(x)

    return result


# %% Full Parameter
# Build a file (YAML?) and call it at the beginning
full_param = {'folder_path': 'C:/Users/riguifd/Desktop/Teste/',
              'file_extension': 'tdms',
              'group_name': 'ACCEL_RAW_DATA', 'channel_name': 'RIG_ACCEL_1D',
              'analysisTime':
                  [{'name': 'BC_CCW', 'start': 1.0, 'end': 4.0},
                   {'name': 'BC_CW', 'start': 5.0, 'end': 8.0},
                   {'name': 'TvS_CCW', 'start': 10.6, 'end': 13.4},
                   {'name': 'Tvs_CW', 'start': 16.2, 'end': 19.0},
                   {'name': 'SR_CCW', 'start': 21.0, 'end': 25.0},
                   {'name': 'SR_CW', 'start': 26.5, 'end': 30.5}],
                  'fft': [{'f_min': 500, 'f_max': 8000, 'plot': True,
                           'smoothing': {'do': True, 'plot': True,
                                         'M': 250, 'p': 0.5, 'sig': 20},
                           'peaks': {'plot': True},
                           'get_top_10': {'print': True}}],
                  'psd': [{'f_min': 500, 'f_max': 8000, 'plot': True,
                           'welch': {'blocksize': 4096, 'window': 'hann',
                                     'overlap': 0.3, 'scaling': 'density',
                                     'average': 'mean'},
                           'smoothing': {'do': True, 'plot': True,
                                         'M': 20, 'p': 0.5, 'sig': 20},
                           'peaks': {'plot': True},
                           'get_top_10': {'print': True}}],
                  'acorr': [{'t_min': 100, 'plot': True,
                             'smoothing': {'do': True, 'plot': True,
                                           'M': 600, 'p': 0.5, 'sig': 600},
                             'peaks': {'plot': True},
                             'get_top_10': {'print': True}}]
              }


# %%
folder = 'C:/Users/riguifd/Desktop/Teste/'
file = list_files('C:/Users/riguifd/Desktop/Teste/', 'tdms')[1]
path = folder + file

sgnl, _, sgnl_param = load_signal_from_tdms(path, 'ACCEL_RAW_DATA','RIG_ACCEL_1D')
sliced_sgnl, sliced_sgnl_param = slicing_by_time(sgnl, sgnl_param, [21.0, 25.0])
get_fft_peaks(sliced_sgnl, sgnl_param, full_param['fft'])
get_psd_peaks(sliced_sgnl, sgnl_param, full_param['psd'])
get_acorr_peaks(sliced_sgnl, sliced_sgnl_param, full_param['acorr'])


# %%
