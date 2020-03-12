# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:10:15 2020

@author: RiguiFD
"""


import file_manager as fm
import signal_processing as sp
import pandas as pd


# Open Parameter File. This 'FullParam' dictionary has most of the Parameters
# That shall be used for NVH_ML

full_param_path = 'C:\\Users\\riguifd\\Documents\\GitHub\\NVH_ML\\NVH_ML\\param.yaml'

main_param = fm.load_yaml_file(full_param_path)


# Get All TDMS file names and entire file path from target directory
list_of_files = fm.list_files(main_param['list_files'])

folder_path = main_param['list_files']['folder_path']
list_of_file_paths = [folder_path + file for file in list_of_files]

full_result = pd.DataFrame()

for file_path in list_of_file_paths:
    print(file_path)
    sgnl, time, sgnl_param = fm.load_signal_from_tdms(file_path,
                                                      main_param['load_signal_from_tdms'])
    print(sgnl_param['f_name'])
    result = pd.DataFrame({'file_name': [sgnl_param['f_name']]})
    for test in main_param['analysis']:
        print(test['test_name'])
        slcd_sgnl, slcd_time, slcd_sgnl_par = fm.slicing_by_time(sgnl, time, sgnl_param, test['time'])
        stat = sp.stats_summ(slcd_sgnl, slcd_time, slcd_sgnl_par, test['stat'], test['test_name'])
        result = result.join(stat)
        x, y, z = sp.get_psd_peaks(slcd_sgnl, slcd_sgnl_par, test['psd'], test['test_name'])

    full_result = full_result.append(result)

# %%Testes de implementação
# sgnl, time, sgnl_param = fm.load_signal_from_tdms(list_of_file_paths[0],
                                                    # main_param['load_signal_from_tdms'])

# slcd_sgnl, slcd_time, slcd_sgnl_par = fm.slicing_by_time(sgnl, time, sgnl_param, main_param['analysis'][0]['time'])

# result = sp.stats_summ(slcd_sgnl, slcd_time, slcd_sgnl_par, main_param['analysis'][0]['stat'], "BC_CCW")

psd_f_val, psd_val, fd = sp.get_psd_values(slcd_sgnl, slcd_sgnl_par, main_param['analysis'][0]['psd'])
trd_sgnl = sp.smoothing_to_get_peaks(psd_val, psd_f_val, main_param['analysis'][0]['psd']['smoothing'])
peaks = sp.get_peaks(trd_sgnl, psd_f_val, int(300/fd), int(21/fd),
                     main_param['analysis'][0]['psd']['peaks'])
get_top_n(trd_sgnl, peaks, fd, main_param['analysis'][0]['psd']['cutting_freq']['f_min'],
          main_param['analysis'][0]['psd']['get_top_n'], main_param['analysis'][0]['test_name'])