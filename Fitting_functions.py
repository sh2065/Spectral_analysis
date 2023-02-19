# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:24:47 2020

@author: sh2065
"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import os,sys
from lmfit.models import VoigtModel, GaussianModel, LorentzianModel, ExponentialModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel, LinearModel, PolynomialModel
from nplab.analysis.background_removal import Adaptive_Polynomial as AP
from scipy import interpolate, signal
import tkinter as tk
from scipy.signal import find_peaks_cwt, find_peaks
import time
import tables



def SI_fitting(laser=633):

    data_path=[]
    laser_power=[]
    File_path_silicon=[]
    File_path=[]
    offset=[]
    peak_int_output=[]
    window = tk.Tk()
    window.title('Silicon band fitting')
    window.geometry('400x600')

    #Creating a frame
    fm1=tk.Frame(window)
    fm1.grid()

    label1=tk.Label(fm1, text='File path SI: ', font=('Arial', 14), height=1,pady=50,padx=10)
    label1.grid(row=0, column=0, sticky='w')
    e1=tk.Entry(fm1, show=None, font=('Arial', 16))
    e1.grid(row=0, column=1, sticky='w')
    
    label2=tk.Label(fm1, text='Data path: ',font=('Arial', 14), height=1,pady=50,padx=10)  
    label2.grid(row=1, column=0, sticky='w') 
    e2=tk.Entry(fm1, show=None, font=('Arial', 16))
    e2.grid(row=1, column=1, sticky='w')

    label3=tk.Label(fm1, text='Laser power: ', font=('Arial', 14), height=1,pady=50,padx=10)
    label3.grid(row=2, column=0, sticky='w')
    e3=tk.Entry(fm1, show=None, font=('Arial', 16))
    e3.grid(row=2, column=1, sticky='w')

    def insert_File_Path_0():
        
        a=e1.get()
        File_path_silicon.append(a)
    
    def insert_Data_path():
        
        b=e2.get()
        data_path.append(b)
    
    def insert_laser_power():
        
        b=e3.get()
        laser_power.append(b)

    def Silicon_offset():
        insert_File_Path_0()
        insert_Data_path()
        insert_laser_power()
        
        
        for i,j in enumerate(File_path_silicon):
            Path_0=j
            File_SI=h5py.File(Path_0, 'r')

        data=np.array(File_SI[str(data_path[0])]).astype(np.float64)
        if len(data.shape) > 1:
            max_int=[]
            for i in data:
                max_int.append(max(i))
            index=np.argmax(max_int)
            data=data[index]
        Wav=np.array(File_SI[str(data_path[0])].attrs['wavelengths']).astype(np.float64)
        if Wav[1]<Wav[0]:
            Wav=np.flipud(Wav)
            data=np.flipud(data)
        Ramanshift=((10.**7)/laser)-((10.**7)/Wav)
        index_520=np.argmax(data)

        data=data[index_520-20:index_520+20]
        Ramanshift=Ramanshift[index_520-20:index_520+20]
        plt.plot(Ramanshift,data)
        data=AP.Run(data, 1 ,Max_Steps=0.5)

        Ramanshift_new=np.linspace(np.min(Ramanshift),np.max(Ramanshift),400)
        y=interpolate.interp1d(Ramanshift,data,kind='quadratic')
        data_new=y(Ramanshift_new)
    
    
    
        GM=GaussianModel()
        Pars=GM.guess(data_new,x=Ramanshift_new)
        res=GM.fit(data_new,Pars,x=Ramanshift_new)
     
        Pars.update(GM.make_params())
            
        resonance = res.params['center'].value
        stderr = res.params['center'].stderr
        fwhm = res.params['fwhm'].value
        sigma = res.params['sigma'].value
        output = res.fit_report(min_correl=0.25)
        fig, ax = plt.subplots()
        #plt.plot(Ramanshift,data)
        plt.plot(Ramanshift_new,data_new)
        plt.plot(Ramanshift_new,res.best_fit)
        #plt.plot(Ramanshift_new,y(Ramanshift_new))
    
        p_index=np.argmax(res.best_fit)
        peak_pos=Ramanshift_new[p_index]
        peak_int=res.best_fit[p_index]
        peak_int_2=max(data_new)
        print(output)

        offset.append(520.6-peak_pos)
        peak_int=peak_int_2/float(np.asarray(laser_power))
        peak_int_output.append(peak_int)
        output_1=['offset:', offset[0], 'peak intensity (mW/s):', peak_int_output[0], 'laser power (mW):', laser_power[0]]
        np.savetxt('offset,peak_int,laser power.txt', output_1, fmt='%s')
        tables.file._open_files.close_all()
        File_SI.close()

        global window
        window = tk.Tk()
        window.title('My window')
        window.geometry('800x300')
        l = tk.Label(window, text=('Peak_int (/mW/s):',peak_int), bg='royalblue', fg='white', font=('Arial', 18), width=30, height=2)
        l.pack(expand=1)
        l2 = tk.Label(window, text=('offset:',offset), bg='royalblue', fg='white', font=('Arial', 18), width=30, height=2)
        l2.pack(expand=1,pady=3)
        window.mainloop() 

    but=tk.Button(window, text='Start', font=('Arial', 14), width=10, height=2, command=Silicon_offset)
    but.grid(row=3, column=0,padx=10)

    window.mainloop()


def Gaussian_fit(data,x,check=False,R_TH=0.5):
    
    if len(x) <= 100:
        x_new=np.linspace(np.min(x),np.max(x),400)
        y=interpolate.interp1d(x,data,kind='quadratic')
        data_new=y(x_new)
        x=x_new.copy()
        data=data_new.copy()
    else:
        pass
    gauss = GaussianModel(prefix='g_')
    pars=gauss.guess(data,x)
    pars.update(gauss.make_params())
    pars['g_center'].set(value=x[np.argmax(data)])
    # pars['g_sigma'].set(value=10,min=6,max=15)
    mod = gauss
    
    init = mod.eval(pars, x=x)
    out = mod.fit(data, pars, x=x)
    
    resonance = out.params['g_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm = out.params['g_fwhm'].value
    height = out.params['g_height'].value
    sigma =out.params['g_sigma'].value

    correlation_matrix = np.corrcoef(data, out.best_fit)
    correlation = correlation_matrix[0,1]
    R_squared = correlation**2
    if R_squared < R_TH:
        print('Fitting fails!')
        print('R_squared is ' +str(R_squared))
        resonance, height, fwhm = [0,0,0]
    if check == True:
        out.plot_fit(datafmt='-')

    return resonance, height, fwhm, x, out.best_fit , R_squared

def Lorentzian_fit(data,x,check=False, R_TH = 0.2):

    x_new=np.linspace(np.min(x),np.max(x),400)
    y=interpolate.interp1d(x,data,kind='quadratic')
    data_new=y(x_new)
    data=data_new.copy()
    x=x_new.copy()
    
    gauss1 =GaussianModel(prefix='g1_')
    pars = gauss1.guess(data, x=x)

    pars.update(gauss1.make_params())
    pars['g1_center'].set(value=x[np.argmax(data)],min=x[np.argmax(data)]-10,max=x[np.argmax(data)]+10)
    pars['g1_sigma'].set(value=5,min=3,max=8)
    #pars['g1_amplitude'].set(value=3.14*np.max(data),min=0.2*3.14*np.max(data)

    mod = gauss1
    init = mod.eval(pars, x=x)
    out = mod.fit(data, pars, x=x)
    comps=out.eval_components(x=x)
    
    resonance_p1 = out.params['g1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm_p1 = out.params['g1_fwhm'].value
    height_p1 = out.params['g1_height'].value
    sigma_p1=out.params['g1_sigma'].value
    p1_fit=comps['g1_']

    correlation_matrix = np.corrcoef(data, out.best_fit)
    correlation = correlation_matrix[0,1]
    R_squared = correlation**2

    if check:
        plt.subplots()
        plt.plot(x,data)
        plt.plot(x_new,out.best_fit)
        print(R_squared )

    # if R_squared < R_TH or out.success == False:
    #     raise Exception('Fitting fails or R-square is below 0.2 !')

    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])

    return Peak_1, out.best_fit, x, R_squared


def Lorentzian_fit_flare(data,x,check=False, R_TH = 0.2):

    
    gauss1 =LorentzianModel(prefix='g1_')
    pars = gauss1.guess(data, x=x)

    pars.update(gauss1.make_params())
    pars['g1_center'].set(value=x[np.argmax(data)],min=630,max=800)
    pars['g1_sigma'].set(value=20,min=10,max=40)
    #pars['g1_amplitude'].set(value=3.14*np.max(data),min=0.2*3.14*np.max(data)

    mod = gauss1
    init = mod.eval(pars, x=x)
    out = mod.fit(data, pars, x=x)
    comps=out.eval_components(x=x)
    
    resonance_p1 = out.params['g1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm_p1 = out.params['g1_fwhm'].value
    height_p1 = out.params['g1_height'].value
    sigma_p1=out.params['g1_sigma'].value
    amp_p1=out.params['g1_amplitude'].value
    p1_fit=comps['g1_']

    correlation_matrix = np.corrcoef(data, out.best_fit)
    correlation = correlation_matrix[0,1]
    R_squared = correlation**2

    if check:
        plt.subplots()
        plt.plot(x,data)
        plt.plot(x,out.best_fit)
        print(R_squared )

    # if R_squared < R_TH or out.success == False:
    #     raise Exception('Fitting fails or R-square is below 0.2 !')

    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])

    return resonance_p1, height_p1, fwhm_p1, sigma_p1,amp_p1, out.best_fit

def Gaussian_fit_exp(data,x,show_fitting=False):

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x)

    gauss = GaussianModel(prefix='g1_')
    pars.update(gauss.make_params())
    max_0=x[np.argmax(abs(data))]
    pars['g1_center'].set(value=x[np.argmax(abs(data))],min=x[np.argmax(abs(data))]-0.01,max=x[np.argmax(abs(data))]+0.01)
    pars['g1_sigma'].set(value=0.02,min=0.01)
    
    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    pars['g2_center'].set(value=max_0+0.01)
    #pars['g2_sigma'].set(value=0.001,min=0.005,max=0.01)
    pars['g2_amplitude'].set(value=1, min=0.1)

    mod=exp_mod + gauss + gauss2

    init = mod.eval(pars, x=x)
    out = mod.fit(data, pars, x=x)
    comps=out.eval_components(x=x)
    
    resonance = out.params['g1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm = out.params['g1_fwhm'].value
    height = out.params['g1_height'].value
    sigma =out.params['g1_sigma'].value

    p1_fit=comps['exp_']
    p2_fit=comps['g1_']
    p3_fit=comps['g2_']


    if show_fitting==True:
        plt.plot(x,data)
        plt.plot(x,p1_fit,'--')
        plt.plot(x,p2_fit,'--')
        plt.plot(x,p3_fit,'--')

    return resonance,height,fwhm,out.best_fit


def DLorentzian_fit(data,x,R_TH=0.5, P1_para=[0,0,5],P2_para=[5,0,4],check=False):

    x_new=np.linspace(np.min(x),np.max(x),400)
    y=interpolate.interp1d(x,data,kind='quadratic')
    data_new=y(x_new)
    
    gauss1 =LorentzianModel(prefix='g1_')
    pars = gauss1.guess(data_new, x=x_new)

    pars.update(gauss1.make_params())
    pars['g1_center'].set(value=x_new[np.argmax(data_new)],min=x_new[np.argmax(data_new)]-10,max=x_new[np.argmax(data_new)]+10)
    pars['g1_sigma'].set(value=P1_para[2],min=P1_para[2]-2,max=P1_para[2]+2)
    #pars['g1_amplitude'].set(value=3.14*np.max(data),min=0.2*3.14*np.max(data))
    
    
    gauss2 = LorentzianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    pars['g2_center'].set(value=x_new[np.argmax(data_new)]+P2_para[0],min=x_new[np.argmax(data_new)]+P2_para[0]-5,max=x_new[np.argmax(data_new)]+P2_para[0]+5)
    pars['g2_sigma'].set(value=P2_para[2],min=P2_para[2]-2,max=P2_para[2]+2)

    mod = gauss1 +gauss2
    
    init = mod.eval(pars, x=x_new)
    out = mod.fit(data_new, pars, x=x_new)
    comps=out.eval_components(x=x_new)
    
    resonance_p1 = out.params['g1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm_p1 = out.params['g1_fwhm'].value
    height_p1 = out.params['g1_height'].value
    sigma_p1=out.params['g1_sigma'].value
    p1_fit=comps['g1_']
    
    resonance_p2 = out.params['g2_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p2 = out.params['g2_fwhm'].value
    height_p2 = out.params['g2_height'].value
    sigma_p2=out.params['g2_sigma'].value
    p2_fit=comps['g2_']

    correlation_matrix = np.corrcoef(data_new, out.best_fit)
    correlation = correlation_matrix[0,1]
    R_squared = correlation**2
    if check:
        plt.subplots()
        plt.plot(x,data)
        plt.plot(x_new,out.best_fit)
        print(R_squared )

    if R_squared < R_TH or out.success == False:
        raise Exception('Fitting fails or R-square is below 0.5 !')
    
    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    #plt.plot(x,data)
    #plt.plot(x,out.best_fit)
    #plt.plot(x,p1_fit)
    #plt.plot(x,p2_fit)

    return Peak_1, Peak_2, out.best_fit, x_new, R_squared

def TLorentzian_fit_DF(data,x,R_TH=0.5,check=False):

    x_new=np.linspace(np.min(x),np.max(x),400)
    y=interpolate.interp1d(x,data,kind='quadratic')
    data_new=y(x_new)

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x)
    gauss1 =GaussianModel(prefix='g1_')
    # pars = gauss1.update(data_new, x=x_new)
    i_0=np.argmin(abs(2.3-x_new))
    i_1=np.argmin(abs(2.45-x_new))

    max_pos=x[np.argmax(data)]
    pars.update(gauss1.make_params())
    if max_pos < 1.85 and np.max(data)/np.max(data_new[i_0:i_1]) > 8:
        pars['g1_center'].set(value=max_pos,min=1.6,max=1.9)
    else:
        pars['g1_center'].set(value=1.8,min=1.67,max=1.96)

    pars['g1_sigma'].set(value=0.05,min=0.02,max=0.1)
    # pars['g1_amplitude'].set(min=0,max=0.2)
    # pars['g1_height'].set(min=0)
    
    
    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    if max_pos < 1.85 and np.max(data)/np.max(data_new[i_0:i_1]) > 8:
        pars.add('delta_center', value=0.28, min=0.05, vary=True)
        pars['g2_center'].set(expr = 'g1_center + delta_center')
    else:
        pars['g2_center'].set(value=2.05,min=1.8,max=2.15)
    pars['g2_sigma'].set(value=0.05,min=0.02,max=0.1)
    # pars.add('delta_sigma', value=0.001, min=0, vary=True)
    # pars['g2_sigma'].set(expr = 'g1_sigma - delta_sigma')
    # amp_v_2=(np.max(data_new[i_2:i_3]))*0.05*3.14
    # pars['g2_amplitude'].set(value=amp_v_2,min=amp_v_2*0.2,max=amp_v_2*2)
    pars.add('g2_height', value=0.01, min=0, max=0.6)

    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())
    center_v=x_new[np.argmax(data_new[i_0:i_1])+i_0]
    pars['g3_center'].set(value=center_v,min=center_v+0.01,max=center_v-0.01)
    pars['g3_sigma'].set(value=0.05,min=0.02,max=0.1)
    amp_v=(np.max(data_new[i_0:i_1]))*0.05*3.14
    pars['g3_amplitude'].set(value=amp_v,min=amp_v*0.98,max=amp_v*1.001)

    mod = exp_mod + gauss1 +gauss2 + gauss3
    
    init = mod.eval(pars, x=x_new)
    out = mod.fit(data_new, pars, x=x_new)
    comps=out.eval_components(x=x_new)
    
    resonance_p1 = out.params['g1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm_p1 = out.params['g1_fwhm'].value
    height_p1 = out.params['g1_height'].value
    sigma_p1=out.params['g1_sigma'].value
    p1_fit=comps['g1_']
    
    resonance_p2 = out.params['g2_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p2 = out.params['g2_fwhm'].value
    height_p2 = out.params['g2_height'].value
    sigma_p2=out.params['g2_sigma'].value
    p2_fit=comps['g2_']
    
    p3_fit=comps['g3_']
    
    p4_fit=comps['exp_']

    correlation_matrix = np.corrcoef(data_new, out.best_fit)
    correlation = correlation_matrix[0,1]
    R_squared = correlation**2
    
    if check:
        plt.subplots()
        plt.plot(x,data)
        plt.plot(x_new,out.best_fit)
        plt.plot(x_new,p1_fit,'--')
        plt.plot(x_new,p2_fit,'--')
        plt.plot(x_new,p3_fit,'--')
        plt.plot(x_new,p4_fit,'--')

        print(R_squared )

    if R_squared < R_TH or out.success == False:
        print('Fitting fails or R-square is below 0.5 !')
    
    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    #plt.plot(x,data)
    #plt.plot(x,out.best_fit)
    #plt.plot(x,p1_fit)
    #plt.plot(x,p2_fit)

    return Peak_1, Peak_2, out.best_fit, x_new, R_squared


def TLorentzian_fit_DF_v2(data,x,R_TH=0.5,check=False):

    x_new=np.linspace(np.min(x),np.max(x),400)
    y=interpolate.interp1d(x,data,kind='quadratic')
    data_new=y(x_new)

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x)

    gauss1 =GaussianModel(prefix='g1_')
    # pars = gauss1.update(data_new, x=x_new)

    i_0=np.argmin(abs(2.3-x_new))
    i_1=np.argmin(abs(2.45-x_new))
    max_pos=x[np.argmax(data)]
    pars.update(gauss1.make_params())
    if max_pos < 1.75:
        pars['g1_center'].set(value=max_pos,min=1.6,max=1.9)
    else:
        pars['g1_center'].set(value=1.8,min=1.67,max=1.96)
    pars['g1_sigma'].set(value=0.05,min=0.02,max=0.1)
    # pars['g1_amplitude'].set(min=0,max=10)
    # pars['g1_height'].set(min=0)
    
    
    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    if max_pos < 1.75:
        pars.add('delta_center', value=0.2, min=0.05, vary=True)
        pars['g2_center'].set(expr = 'g1_center + delta_center')
    elif max_pos > 2.0:
        pars['g2_center'].set(value=max_pos,min=1.8,max=2.15)
        pars['g2_height'].set(value=0.5*np.max(data),min=0)
        pars.add('delta_center', value=0.3, min=0.05, vary=True)
        pars['g1_center'].set(expr = 'g2_center - delta_center')
    else:
        pars['g2_center'].set(value=2.05,min=1.8,max=2.15)
        # pars['g2_amplitude'].set(min=0,max=10)

    pars['g2_sigma'].set(value=0.05,min=0.02,max=0.1)
    # pars.add('delta_sigma', value=0.001, min=0, vary=True)
    # pars['g2_sigma'].set(expr = 'g1_sigma - delta_sigma')
    # pars['g2_amplitude'].set(value=1/3.14,min=0)

    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())
    center_v=x_new[np.argmax(data_new[i_0:i_1])+i_0]
    pars['g3_center'].set(value=center_v,min=center_v+0.01,max=center_v-0.01)
    pars['g3_sigma'].set(value=0.05,min=0.02,max=0.1)
    amp_v=(np.max(data_new[i_0:i_1]))*0.05*3.14
    pars['g3_amplitude'].set(value=amp_v,min=amp_v*0.98,max=amp_v*1.001)

    mod = exp_mod + gauss1 +gauss2 + gauss3
    
    init = mod.eval(pars, x=x_new)
    out = mod.fit(data_new, pars, x=x_new)
    comps=out.eval_components(x=x_new)
    
    resonance_p1 = out.params['g1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm_p1 = out.params['g1_fwhm'].value
    height_p1 = out.params['g1_height'].value
    sigma_p1=out.params['g1_sigma'].value
    p1_fit=comps['g1_']
    
    resonance_p2 = out.params['g2_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p2 = out.params['g2_fwhm'].value
    height_p2 = out.params['g2_height'].value
    sigma_p2=out.params['g2_sigma'].value
    p2_fit=comps['g2_']
    
    p3_fit=comps['g3_']
    
    p4_fit=comps['exp_']

    correlation_matrix = np.corrcoef(data_new, out.best_fit)
    correlation = correlation_matrix[0,1]
    R_squared = correlation**2
    
    if check:
        plt.subplots()
        plt.plot(x,data)
        plt.plot(x_new,out.best_fit)
        plt.plot(x_new,p1_fit,'--')
        plt.plot(x_new,p2_fit,'--')
        plt.plot(x_new,p3_fit,'--')
        plt.plot(x_new,p4_fit,'--')

        print(R_squared )

    if R_squared < R_TH or out.success == False:
        print('Fitting fails or R-square is below 0.5 !')
    
    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    #plt.plot(x,data)
    #plt.plot(x,out.best_fit)
    #plt.plot(x,p1_fit)
    #plt.plot(x,p2_fit)

    return Peak_1, Peak_2, out.best_fit, x_new, R_squared

def TLorentzian_fit_DF_cube(data,x,R_TH=0.5,check=False):

    x_new=np.linspace(np.min(x),np.max(x),400)
    y=interpolate.interp1d(x,data,kind='quadratic')
    data_new=y(x_new)

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x)
    gauss1 =GaussianModel(prefix='g1_')
    # pars = gauss1.update(data_new, x=x_new)
    i_0=np.argmin(abs(2.25-x_new))
    i_1=np.argmin(abs(2.45-x_new))

    max_pos=x[np.argmax(data)]
    pars.update(gauss1.make_params())
    # if max_pos < 1.75 and np.max(data)/np.max(data_new[i_0:i_1]) > 8:
    #     pars['g1_center'].set(value=max_pos,min=1.6,max=1.9)
    # else:
    pars['g1_center'].set(value=1.6,min=1.5,max=1.85)

    pars['g1_sigma'].set(value=0.05,min=0.02,max=0.1)
    # pars['g1_amplitude'].set(min=0,max=0.2)
    # pars['g1_height'].set(min=0)
    
    
    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    # if max_pos < 1.75 and np.max(data)/np.max(data_new[i_0:i_1]) > 8:
    #     pars.add('delta_center', value=0.28, min=0.05, vary=True)
    #     pars['g2_center'].set(expr = 'g1_center + delta_center')
    # else:
    pars['g2_center'].set(value=1.8,min=1.7,max=2.0)
    pars['g2_sigma'].set(value=0.05,min=0.02,max=0.1)
    # pars.add('delta_sigma', value=0.001, min=0, vary=True)
    # pars['g2_sigma'].set(expr = 'g1_sigma - delta_sigma')
    # amp_v_2=(np.max(data_new[i_2:i_3]))*0.05*3.14
    # pars['g2_amplitude'].set(value=amp_v_2,min=amp_v_2*0.2,max=amp_v_2*2)
    pars.add('g2_height', value=0.01, min=0, max=0.6)

    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())
    center_v=x_new[np.argmax(data_new[i_0:i_1])+i_0]
    pars['g3_center'].set(value=center_v,min=center_v+0.01,max=center_v-0.01)
    pars['g3_sigma'].set(value=0.07,min=0.03,max=0.15)
    amp_v=(np.max(data_new[i_0:i_1]))*0.07*3.14
    pars['g3_amplitude'].set(value=amp_v,min=amp_v*0.98,max=amp_v*1.001)

    mod =  gauss1 +gauss2 + gauss3
    
    init = mod.eval(pars, x=x_new)
    out = mod.fit(data_new, pars, x=x_new)
    comps=out.eval_components(x=x_new)
    
    resonance_p1 = out.params['g1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm_p1 = out.params['g1_fwhm'].value
    height_p1 = out.params['g1_height'].value
    sigma_p1=out.params['g1_sigma'].value
    p1_fit=comps['g1_']
    
    resonance_p2 = out.params['g2_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p2 = out.params['g2_fwhm'].value
    height_p2 = out.params['g2_height'].value
    sigma_p2=out.params['g2_sigma'].value
    p2_fit=comps['g2_']
    
    p3_fit=comps['g3_']
    
    # p4_fit=comps['exp_']

    correlation_matrix = np.corrcoef(data_new, out.best_fit)
    correlation = correlation_matrix[0,1]
    R_squared = correlation**2
    
    if check:
        plt.subplots()
        plt.plot(x,data)
        plt.plot(x_new,out.best_fit)
        plt.plot(x_new,p1_fit,'--')
        plt.plot(x_new,p2_fit,'--')
        plt.plot(x_new,p3_fit,'--')
        # plt.plot(x_new,p4_fit,'--')

        print(R_squared )

    if R_squared < R_TH or out.success == False:
        print('Fitting fails or R-square is below 0.5 !')
    
    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    #plt.plot(x,data)
    #plt.plot(x,out.best_fit)
    #plt.plot(x,p1_fit)
    #plt.plot(x,p2_fit)

    return Peak_1, Peak_2, out.best_fit, x_new, R_squared

def TLorentzian_fit_DF_96_sphere(data,x,R_TH=0.5,check=False):

    x_new=np.linspace(np.min(x),np.max(x),400)
    y=interpolate.interp1d(x,data,kind='quadratic')
    data_new=y(x_new)

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x)
    gauss1 =GaussianModel(prefix='g1_')
    # pars = gauss1.update(data_new, x=x_new)
    i_0=np.argmin(abs(2.25-x_new))
    i_1=np.argmin(abs(2.45-x_new))

    max_pos=x[np.argmax(data)]
    pars.update(gauss1.make_params())
    if max_pos < 1.85 and np.max(data)/np.max(data_new[i_0:i_1]) > 8:
        pars['g1_center'].set(value=max_pos,min=1.6,max=1.9)
    else:
        pars['g1_center'].set(value=1.75,min=1.67,max=1.96)

    pars['g1_sigma'].set(value=0.05,min=0.02,max=0.1)
    # pars['g1_amplitude'].set(min=0,max=0.2)
    # pars['g1_height'].set(min=0)
    
    
    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    if max_pos < 1.85 and np.max(data)/np.max(data_new[i_0:i_1]) > 8:
        pars.add('delta_center', value=0.28, min=0.01, vary=True)
        pars['g2_center'].set(expr = 'g1_center + delta_center')
    else:
        pars['g2_center'].set(value=2.05,min=1.8,max=2.15)
    pars['g2_sigma'].set(value=0.05,min=0.02,max=0.1)
    # pars.add('delta_sigma', value=0.001, min=0, vary=True)
    # pars['g2_sigma'].set(expr = 'g1_sigma - delta_sigma')
    # amp_v_2=(np.max(data_new[i_2:i_3]))*0.05*3.14
    # pars['g2_amplitude'].set(value=amp_v_2,min=amp_v_2*0.2,max=amp_v_2*2)
    # pars.add('g2_height', value=0.01, min=0, max=0.6)

    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())
    center_v=x_new[np.argmax(data_new[i_0:i_1])+i_0]
    pars['g3_center'].set(value=center_v,min=center_v+0.01,max=center_v-0.01)
    pars['g3_sigma'].set(value=0.05,min=0.02,max=0.1)
    amp_v=(np.max(data_new[i_0:i_1]))*0.05*3.14
    pars['g3_amplitude'].set(value=amp_v,min=amp_v*0.98,max=amp_v*1.001)

    mod = gauss1 +gauss2 + gauss3
    
    init = mod.eval(pars, x=x_new)
    out = mod.fit(data_new, pars, x=x_new)
    comps=out.eval_components(x=x_new)
    
    resonance_p1 = out.params['g1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm_p1 = out.params['g1_fwhm'].value
    height_p1 = out.params['g1_height'].value
    sigma_p1=out.params['g1_sigma'].value
    p1_fit=comps['g1_']
    
    resonance_p2 = out.params['g2_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p2 = out.params['g2_fwhm'].value
    height_p2 = out.params['g2_height'].value
    sigma_p2=out.params['g2_sigma'].value
    p2_fit=comps['g2_']
    
    p3_fit=comps['g3_']
    
    # p4_fit=comps['exp_']

    correlation_matrix = np.corrcoef(data_new, out.best_fit)
    correlation = correlation_matrix[0,1]
    R_squared = correlation**2
    
    if check:
        plt.subplots()
        plt.plot(x,data)
        plt.plot(x_new,out.best_fit)
        plt.plot(x_new,p1_fit,'--')
        plt.plot(x_new,p2_fit,'--')
        plt.plot(x_new,p3_fit,'--')
        # plt.plot(x_new,p4_fit,'--')

        print(R_squared )

    if R_squared < R_TH or out.success == False:
        print('Fitting fails or R-square is below 0.5 !')
    
    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    #plt.plot(x,data)
    #plt.plot(x,out.best_fit)
    #plt.plot(x,p1_fit)
    #plt.plot(x,p2_fit)

    return Peak_1, Peak_2, out.best_fit, x_new, R_squared