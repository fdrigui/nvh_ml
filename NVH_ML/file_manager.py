# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:36:31 2020

@author: RiguiFD
"""


import nptdms
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('ggplot')


def list_files(param):

    files = [fn for fn in os.listdir(param['folder_path'])
             if fn.endswith(param['file_extension'])]
    return files


def load_signal_from_tdms(file_path, param) -> np.ndarray:

    tdms_file = nptdms.TdmsFile(file_path)

    channel = tdms_file.object(param['group_name'], param['channel_name'])

    data = channel.data
    time = channel.time_track()

    file_name = file_path[::-1][:file_path[::-1].find('/')][::-1]
    t_n = time[-1]
    N = data.size
    dt = t_n/N
    fs = int(round(np.reciprocal(dt), 0))
    dt = np.reciprocal(float(fs))
    data_param = {'t_n': t_n, 'N': N, 'dt': dt, 'fs': fs, 'f_name': file_name}

    if param["plot"]["do"]:
        plt.plot(time, data)
        plt.xlabel(param["plot"]["x_label"], fontsize=16)
        plt.ylabel(param["plot"]["y_label"], fontsize=16)
        plt.title(param["plot"]["title"], fontsize=16)
        plt.show()

    return data, time, data_param


def slicing_by_time(sgnl, time, sgnl_param, param):

    start = param['start']
    end = param['end']
    sliced_sgnl = sgnl[int(start*sgnl_param['fs']):int(end*sgnl_param['fs'])]
    sliced_time = time[int(start*sgnl_param['fs']):int(end*sgnl_param['fs'])]

    sgnl_param['N'] = sliced_sgnl.size
    sgnl_param['t_n'] = sgnl.size*sgnl_param['dt']
    sgnl_param['dt'] = sgnl_param['dt']
    sgnl_param['fs'] = sgnl_param['fs']
    sgnl_param['f_name'] = sgnl_param['f_name']

    if param['plot']['do']:
        plt.plot(sliced_time, sliced_sgnl)
        plt.xlabel(param["plot"]["x_label"], fontsize=16)
        plt.ylabel(param["plot"]["y_label"], fontsize=16)
        plt.title(param["plot"]["title"], fontsize=16)
        plt.annotate(sgnl_param['f_name'], (0,0))
        plt.show()

    return sliced_sgnl, sliced_time, sgnl_param


def load_yaml_file(path: str):
    result = {}
    with open(path, 'r') as stream:
        try:
            result = yaml.safe_load(stream)

        except yaml.YAMLError as exc:
            print(exc)
    return result
