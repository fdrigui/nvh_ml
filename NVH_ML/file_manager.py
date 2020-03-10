# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:36:31 2020

@author: RiguiFD
"""


import nptdms
import os
import yaml
import numpy as np


def list_files(param):

    files = [fn for fn in os.listdir(param['folder_path'])
             if fn.endswith(param['file_extension'])]
    return files


def load_signal_from_tdms(file_path: str, group_name: str,
                          channel_name: str) -> np.ndarray:

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


def load_yaml_file(path: str):
    result = {}
    with open(path, 'r') as stream:
        try:
            result = yaml.safe_load(stream)

        except yaml.YAMLError as exc:
            print(exc)
    return result


class NVH_sgnl(object):
    t_n = None
    N = None
    dt = None
    fs = None
    f_name = None
    pass
