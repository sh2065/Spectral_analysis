# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:22:53 2020

@author: sh2065
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from lmfit.models import VoigtModel, GaussianModel, LorentzianModel, ExponentialModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel,  LinearModel, PolynomialModel, SplitLorentzianModel, PseudoVoigtModel
from scipy.optimize import curve_fit
import time
from scipy import interpolate, signal
from scipy.signal import savgol_filter as sgFilt
import os
from nplab.analysis.background_removal import Adaptive_Polynomial as AP
import scipy.stats as stats 
import matplotlib
from nplab.analysis.background_removal import Iterative_Polynomial as IP
from tqdm import tqdm

def butterLowpassFiltFilt(data, cutoff = 1500, fs = 60000, order=1):
    '''Smoothes data without shifting it'''
    nyq = 0.5 * fs
    normalCutoff = cutoff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    yFiltered = filtfilt(b, a, data)
    return yFiltered

def reduceNoise(y, factor = 10, cutoff = 1500, fs = 60000, pl = False):

    if pl == True:
        ySmooth = sgFilt(y, window_length = 221, polyorder = 7)

    ySmooth = butterLowpassFiltFilt(y, cutoff = cutoff, fs = fs)
    yNoise = y - ySmooth
    yNoise /= factor
    y = ySmooth + yNoise
    return y

def Filter(data,threshold=0):
    data_n=[i for i in data if i > threshold]
    data_n=np.asarray(data_n)
    return data_n


def bgd_r (x,y):

    spe_00=y
    Wav=x
    try:
        spe_0=IP.Run(Wav,spe_00,Poly_Order=2,Maximum_Iterations=10)
        if np.min(reduceNoise(spe_0[100:400]))< -0.0003:
            spe_0=IP.Run(Wav,spe_00,Poly_Order=1,Maximum_Iterations=100)
            if np.min(reduceNoise(spe_0[640:])) > np.mean(reduceNoise(spe_0)):
                spe_0=IP.Run(Wav,spe_00,Poly_Order=3,Maximum_Iterations=10)    
                if np.min(reduceNoise(spe_0[100:400]))< -0.001:
                    spe_0=IP.Run(Wav,spe_00,Poly_Order=1,Maximum_Iterations=100)                         
            elif np.max(reduceNoise(spe_0)) - np.mean(reduceNoise(spe_0[640:])) < 0.002 and np.mean(reduceNoise(spe_0[100:300]))<np.mean(reduceNoise(spe_0[640:])):
                spe_0=IP.Run(Wav,spe_00,Poly_Order=3,Maximum_Iterations=10)    
                if np.min(reduceNoise(spe_0[100:400]))< -0.001:
                    spe_0=IP.Run(Wav,spe_00,Poly_Order=1,Maximum_Iterations=100)                
            
            elif np.min(reduceNoise(spe_0[100:400]))< -0.0003:
                spe_0=IP.Run(Wav,spe_00,Poly_Order=0,Maximum_Iterations=10)
                if np.min(reduceNoise(spe_0[100:400]))< -0.0003:
                    spe_0=spe_00-np.min(reduceNoise(spe_00[:400]))
                    
                
            elif np.max(reduceNoise(spe_0[640:]))<-0.001:
                spe_0=IP.Run(Wav,spe_00,Poly_Order=2,Maximum_Iterations=10)
                
            elif np.mean(reduceNoise(spe_0[:60]))<-0.0003:
                spe_0=spe_00-np.min(reduceNoise(spe_00[:400]))
        
        elif np.min(reduceNoise(spe_0[640:])) > np.mean(reduceNoise(spe_0)) or np.max(reduceNoise(spe_0)) - np.mean(reduceNoise(spe_0[640:])) < 0.0022:
            spe_0=IP.Run(Wav,spe_00,Poly_Order=3,Maximum_Iterations=10)
            spe_1=IP.Run(Wav,spe_00,Poly_Order=2,Maximum_Iterations=10)
            h_1=abs(np.max(reduceNoise(spe_0[50:200]))-np.max(reduceNoise(spe_1[50:200])))
           
            if np.min(reduceNoise(spe_0[100:400]))< -0.001:
                spe_0=IP.Run(Wav,spe_00,Poly_Order=1,Maximum_Iterations=100)  
                v_0=[]
                for j in spe_0:
                    if j<0:
                        v_0.append(j)                       
                if len(v_0) > 100:
                    spe_0=spe_00-np.min(reduceNoise(spe_00[:400]))  
            elif h_1>0.0005 :
                spe_0=IP.Run(Wav,spe_00,Poly_Order=2,Maximum_Iterations=10)

        h_0=[]
        for z in spe_0:
            if z<0:
                h_0.append(z)                       
        if len(h_0) > 150 or np.mean(reduceNoise(spe_0[640:]))>np.max(reduceNoise(spe_0[300:500])):
            spe_0=spe_00-np.min(reduceNoise(spe_00[:400]))
    except:
        raise Exception ('Werid point!')

    return spe_0

def bgd_r_v2 (x,y):
    i0=np.argmin(abs(x-600))
    i1=np.argmin(abs(x-750))
    i_cut=np.argmin(reduceNoise(y[i0:i1]))+i0
    spe_0=IP.Run(x[:i_cut+100],y[:i_cut+100],Poly_Order=1,Maximum_Iterations=10)
    spe_1=IP.Run(x[i_cut:],y[i_cut:],Poly_Order=1,Maximum_Iterations=10)

    if np.min(spe_1[:50])<0 :
        spe_1=spe_1-np.min(spe_1[:50])

    if np.min(spe_0[-150:-100])<0 :
        spe_0=spe_0-np.min(spe_0[-150:-100])

    spe=np.concatenate((spe_0[:-100],spe_1),axis=0)
    return spe

'''
a=f_1.create_group('NPoMs (Raw and fitted spectra)')
b=f_1.create_group('NPoMs (Weird spectra)')
aaa=f_1.create_group('NPoMs (Bg subtracted and fitted spectra)')

for i in tqdm(range(len(Spe))):
    try:
        Peak_1_0,Peak_2_0,Peak_3_0, best_fit= Multi_Peaks_fit_Gau(Spe[i],Wav_new)
        fit=0
    except:
        Peak_1_0,Peak_2_0,Peak_3_0, best_fit= Multi_Peaks_fit_Gau(Spe[i][:-10],Wav_new[:-10])
        fit=1

    #fun_evals_0=Gaussianfit(Spe[i],Wav_new)
    #fun_evals.append(int(fun_evals_0[121:125]))
    spe_sms_0=reduceNoise(Spe[i][:-10])
    peaks=signal.find_peaks(spe_sms_0,width=18)
    if Peak_3_0[2]>=150 or Peak_3_0[1] <= 0.0006 :
        try:
            data_0=np.concatenate((Spe[i],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit)))
        except:
            data_0=np.concatenate((Spe[i][:-10],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit))) 
        Fail_spe.append(Spe[i])
        Fail_name.append(name[i])
        c0=b.create_dataset(name=name[i],data=data_0)
        c0.attrs['wavelengths']=Wav_new

    elif Wav_new[np.argmax(spe_sms_0)] >= 880 and Peak_3_0[0] >= 890:
        try:
            data_0=np.concatenate((Spe[i],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit)))
        except:
            data_0=np.concatenate((Spe[i][:-10],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit))) 
        Fail_spe.append(Spe[i])
        Fail_name.append(name[i])
        c0=b.create_dataset(name=name[i],data=data_0)
        c0.attrs['wavelengths']=Wav_new
    elif np.max(best_fit)-Peak_3_0[1] >= 0.004 and Wav_new[np.argmax(best_fit)] > 900:
        try:
            data_0=np.concatenate((Spe[i],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit)))
        except:
            data_0=np.concatenate((Spe[i][:-10],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit))) 
        Fail_spe.append(Spe[i])
        Fail_name.append(name[i])
        c0=b.create_dataset(name=name[i],data=data_0)
        c0.attrs['wavelengths']=Wav_new

    elif len(peaks[0]) >=5 :
        try:
            data_0=np.concatenate((Spe[i],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit)))
        except:
            data_0=np.concatenate((Spe[i][:-10],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit)))
        Fail_spe.append(Spe[i])
        Fail_name.append(name[i])
        c0=b.create_dataset(name=name[i],data=data_0)
        c0.attrs['wavelengths']=Wav_new

    elif  abs(Wav_new[np.argmax(spe_sms_0)]-Peak_3_0[0])>=12 and abs(np.max(best_fit)-np.max(spe_sms_0)) >0.05:
        try:
            data_0=np.concatenate((Spe[i],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit)))
        except:
            data_0=np.concatenate((Spe[i][:-10],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit))) 
        Fail_spe.append(Spe[i])
        Fail_name.append(name[i])
        c0=b.create_dataset(name=name[i],data=data_0)   
        c0.attrs['wavelengths']=Wav_new

    else:
        if len(best_fit) != len(Spe[i]):
            
            data_0=np.concatenate((Spe[i][:-10],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit)))
            
            data_000=bgd_r(Wav_new,Spe[i])
            
            Peak_1_0,Peak_2_0,Peak_3_0, best_fit= Multi_Peaks_fit_Gau(data_000[:-10],Wav_new[:-10])

            data_0000=np.concatenate((Spe[i][:-10],data_000[:-10],best_fit),axis=0)
            data_0000=np.reshape(data_0000,(-1,len(data_000[:-10])))
            c1=a.create_dataset(name=name[i],data=data_0)
            c1.attrs['wavelengths']=Wav_new
            
            c11=aaa.create_dataset(name=name[i],data=data_0000)
            c0.attrs['wavelengths']=Wav_new

            Peak_1.append(Peak_1_0)
            Peak_2.append(Peak_2_0)
            Peak_3.append(Peak_3_0)
            Spe_succ.append(Spe[i])

        else:
            data_0=np.concatenate((Spe[i],best_fit),axis=0)
            data_0=np.reshape(data_0,(-1,len(best_fit)))
            
            data_000=bgd_r(Wav_new,Spe[i])
           
            Peak_1_0,Peak_2_0,Peak_3_0, best_fit= Multi_Peaks_fit_Gau(data_000,Wav_new)
            
            data_0000=np.concatenate((Spe[i],data_000,best_fit),axis=0)
            data_0000=np.reshape(data_0000,(-1,len(data_000)))
            c1=a.create_dataset(name=name[i],data=data_0)
            c1.attrs['wavelengths']=Wav_new
            
            c11=aaa.create_dataset(name=name[i],data=data_0000)
            c11.attrs['wavelengths']=Wav_new

            Peak_1.append(Peak_1_0)
            Peak_2.append(Peak_2_0)  
            Peak_3.append(Peak_3_0)   
            Spe_succ.append(Spe[i])
        
        Succ_name.append(name[i])

    if i+1 == int(len(Spe)):
        print('All finished! (100%)')
'''

