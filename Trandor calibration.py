# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:41:40 2023

@author: sh2065
"""

import numpy as np
import matplotlib.pyplot as plt
from Shu_analysis import Data_processing as DP
from Shu_analysis import DF_sorting as DFS



File_path_0=r'D:\Data analysis_Extension\LAB5-DATA\20220725_two wavelength SERS 81 NDoM_circular polarized light\2022-07-25.h5'
File_path_1 = r'W:\Data\sh2065\LAB5-DATA\20231108_PbS QDs from MIT\2023-11-08.h5'
data_path_0= ['/OceanOpticsSpectrometer/room light_2']
data_path_1= ['/AndorData/room light_for_cal']


Wav_cal, res = DP.td_calibration (File_path=[File_path_0, File_path_1], data_path=[data_path_0, data_path_1], spe_range=[550,1050], peaks_oo_remove=[627,973,1012], check=True)

