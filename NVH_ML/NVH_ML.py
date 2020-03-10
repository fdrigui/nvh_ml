# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:10:15 2020

@author: RiguiFD
"""


import file_manag as fm



# Open Parameter File. This 'FullParam' dictionary has most of the Parameters
# That shall be used for NVH_ML

full_param_path = 'C:\\Users\\riguifd\\Documents\\Python Scripts\\NVH_Analysis\\param.yaml'
main_param = fm.load_yaml_file(full_param_path)


# Get All TDMS file names and entire file path from target directory

list_of_files = fm.list_files(main_param['list_files']['folder_path'],
                              main_param['list_files']['file_extension'])

list_of_file_paths = [main_param['list_files']['folder_path'] + file for file in list_of_files]

for file_path in list_of_file_paths:
    print(file_path)


# %%Testes de implementação

