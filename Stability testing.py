# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:32:46 2023

@author: sh2065
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from scipy.signal import find_peaks
from nplab.analysis.background_removal import Adaptive_Polynomial as AP
from tqdm import tqdm
from lmfit.models import GaussianModel, LorentzianModel
from Spectral_analysis import plt_fig as pf

def DF_spe_sorting(File_path,data_path, x_lim=[480,950], data_type=''):
    with h5py.File(File_path, 'r') as File:
        data=np.asarray(File[data_path])
        bgd=File[data_path].attrs['background']
        ref=File[data_path].attrs['reference']
        Wav=File[data_path].attrs['wavelengths']
        i_0=np.argmin(abs(Wav-x_lim[0]))
        i_1=np.argmin(abs(Wav-x_lim[1]))
        Wav = Wav[i_0:i_1]
        if data_type == 'z_scan':
            data=np.max(data,axis=0)
            spe=(data-bgd)/(ref-bgd)
            spe=spe[i_0:i_1]
        else:
            spe=(data-bgd)/(ref-bgd)
            spe=spe[:,i_0:i_1]
    return Wav, spe

def DF_laser(File_path,data_path,x_lim=[510,950]):
    spe=[]
    for i,j in enumerate(data_path):
        Wav, data=DF_spe_sorting(File_path,j,x_lim, data_type='z_scan')
        spe.append(data)
        if i==0:
            pf.plt_plot(Wav, data*100,label_s=22)
        else:
            plt.plot(Wav, data*100,linewidth=3)
    plt.subplots_adjust(top=0.98,bottom=0.14,left=0.135,right=0.965,hspace=0.2,wspace=0.2)
    plt.xlabel('Wavlength (nm)')
    plt.ylabel('Scattering intensity (%)')
    # return Wav, spe

def DF_time_series(File_path,data_path,x_lim=[510,950], exposure = 10):
    
    Wav, data=DF_spe_sorting(File_path,data_path,x_lim)
    time=np.arange(0,len(data[1:])*10,10)
    for i,j in enumerate(data[1:]):
        if i==0:
            pf.plt_plot(Wav, j*100,label_s=22)
        else:
            plt.plot(Wav, j*100, linewidth=3)
    plt.xlabel('Wavlength (nm)')
    plt.ylabel('Scattering intensity (%)')
    plt.subplots_adjust(top=0.98,bottom=0.14,left=0.165,right=0.965,hspace=0.2,wspace=0.2)
    pf.plt_contourf(Wav,time, data[1:], pcolormesh=True,label_s=20,figsize=[6,9],color_range=[0.6,1])

def Run(File_path, scan_num, particle_num):
    DF_time_series(File_path, '/ParticleScannerScan_'+str(scan_num)+'/Particle_'+str(particle_num)+'/DF_laserpower_15.0')
    DF_laser(File_path, ['/ParticleScannerScan_'+str(scan_num)+'/Particle_'+str(particle_num)+'/z_scan_initial10 s_irr',
                         '/ParticleScannerScan_'+str(scan_num)+'/Particle_'+str(particle_num)+'/z_scan_after10 s_irr'])


File_path_0=r'W:\Data\dm958\20230229_nanocavity stability\2023-03-01.h5'
File_path_1=r'W:\Data\dm958\20230227_nanocavity stability\2023-02-27.h5'

Run(File_path_0, 0, 1)

Run(File_path_0, 0, 3)
Run(File_path_0, 0, 7)
Run(File_path_0, 0, 14)

Run(File_path_1, 0, 3)
Run(File_path_1, 1, 3)

Run(File_path_1, 1, 15)
