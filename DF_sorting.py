# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:19:48 2021

@author: sh2065
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from lmfit.models import VoigtModel, GaussianModel, LorentzianModel, ExponentialModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel,LinearModel, PolynomialModel, SplitLorentzianModel, PseudoVoigtModel
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
import cv2 as cv
from scipy.signal import find_peaks
from Shu_analysis import plt_fig as pf

def File_input():
    try:
        File_path=np.loadtxt('File_path.txt',delimiter=',', dtype=np.str)
        print('Loading File...')
    except:
        File_path=input('Please enter your file path: ')
        Data_name=input('Please enter the name of your data: ')
        np.savetxt('File_path.txt',[File_path, Data_name],fmt='%s')
        File_path=np.loadtxt('File_path.txt',delimiter=',', dtype=np.str)
    return File_path

def Linearfit(data,Wav):
    LM=LinearModel()
    x=Wav
    Pars=LM.guess(data,x=Wav)
    res=LM.fit(data,Pars,x=Wav)
    if len(data)<200:
        x_new=np.linspace(np.min(Wav),np.max(Wav),200)
        y=interpolate.interp1d(x,res.best_fit,kind='quadratic')
        best_fit=y(x_new)
        x=x_new
    else:
        best_fit=res.best_fit
    return best_fit, x

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

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def despike(yi, th=1):
    '''Remove spike from array yi, the spike area is where the difference between 
    the neigboring points is higher than th.'''
    y = np.copy(yi) # use y = y1 if it is OK to modify input array
    n = len(y)
    x = np.arange(n)
    c = np.argmax(y)
    d = abs(np.diff(y))
    try:
      l = c - 1 - np.where(d[c-1::-1]<th)[0][0]
      r = c + np.where(d[c:]<th)[0][0] + 1
    except: # no spike, return unaltered array
      return y
    # for fit, use area twice wider then the spike
    if (r-l) <= 3:
      l -= 1
      r += 1
    s = int(round((r-l)/2.))
    lx = l - s
    rx = r + s
    # make a gap at spike area
    xgapped = np.concatenate((x[lx:l],x[r:rx]))
    ygapped = np.concatenate((y[lx:l],y[r:rx]))
    # quadratic fit of the gapped array
    z = np.polyfit(xgapped,ygapped,2)
    p = np.poly1d(z)
    y[l:r] = p(x[l:r])
    return y


def Gaussian(data,x):
    
    d=np.max(x)-np.min(x)
    x_new=np.linspace(np.min(x),np.max(x),200)
    y=interpolate.interp1d(x,data,kind='quadratic')
    data_new=y(x_new)   
    
    x_new_1=np.linspace(np.min(x)-0.25*d,np.max(x)+0.25*d,300)
    data_new_ext_1=[]
    data_new_ext_2=[]
    for i in x_new_1:
        if i<np.min(x):
            data_new_ext_1.append(0)
        if i>np.max(x):
            data_new_ext_2.append(0)
    data_new_ext_1=np.asarray(data_new_ext_1)
    data_new_ext_2=np.asarray(data_new_ext_2)
    
    data_new=np.concatenate((data_new_ext_1,data_new),axis=0) 
    data_new=np.concatenate((data_new,data_new_ext_2),axis=0) 
    x_new=x_new_1
   
    GM=GaussianModel(prefix='Gaussian_')   
    pars=GM.guess(data_new,x=x_new)
    
    init = GM.eval(pars, x=x_new)
    out = GM.fit(data_new, pars, x=x_new)
    
    resonance = out.params['Gaussian_center'].value
    stderr = out.params['Gaussian_center'].stderr
    fwhm = out.params['Gaussian_fwhm'].value
    sigma = out.params['Gaussian_sigma'].value
    
    # plt.plot(x_new, data_new)
    # plt.plot(x_new, out.init_fit, 'k--', label='initial fit')
    # plt.plot(x_new, out.best_fit, 'r-', label='best fit')
    # plt.legend(loc='best')
    # plt.show()
    
    return resonance, fwhm, out.best_fit, x_new

def check_center(image,check=False, threshold=10,pixel_range=[40,60],rd_range=[0,16]):

    img_0=np.asarray(image)
    img_cut=img_0[pixel_range[0]:pixel_range[1],pixel_range[0]:pixel_range[1],:]
    imgray = cv.cvtColor(img_cut, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, threshold, 255, cv.THRESH_TOZERO)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    result=False
    for i,j in enumerate (contours):
        (x, y), radius=cv.minEnclosingCircle(contours[i])
        min_0=(pixel_range[1]-pixel_range[0])/2 -5
        max_0=(pixel_range[1]-pixel_range[0])/2 +5
        if rd_range[1] > radius > rd_range[0]  and min_0 < x < max_0 and  min_0 < y < max_0:
            result=True
            break

    if check:
        plt.subplots()
        imgray_w = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY)
        ret_w, thresh_w = cv.threshold(imgray_w, threshold, 255, cv.THRESH_BINARY)
        contours_w, hierarchy_w = cv.findContours(thresh_w, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img_ct=cv.drawContours(img_0, contours_w, -1, (0,255,0), 1)
        plt.imshow(img_ct)
        plt.subplots()
        x, y, radius = np.int0((x,y,radius))
        img_c=cv.circle(img_cut, (x,y), radius, (0, 0, 255), 2)
        plt.imshow(img_c)
        print('Radius is:' +str(radius))
    return result

def Multi_Peaks_fit_Gau_51_Dec(data, Wav, check=False):

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x=Wav)
       
    gauss1_1 = GaussianModel(prefix='g1_1_')
    pars.update(gauss1_1.make_params())

    pars['g1_1_center'].set(value=530, min=510, max=550)
    pars['g1_1_sigma'].set(value=18,min=17,max=19)
    amp_g1_1=np.max(data[:90])*18*2.5*0.4
    pars['g1_1_amplitude'].set(value=amp_g1_1, min=amp_g1_1*0.95,max=amp_g1_1*1.5)
    
    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    pars['g1_center'].set(value=540, min=530, max=590)
    pars['g1_sigma'].set(value=20,min=19,max=21)
    amp_g1=np.max(data[:180])*20*2.5*0.6
    pars['g1_amplitude'].set(value=amp_g1, min=amp_g1*0.9, max=amp_g1*1.5)

    
    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    pars['g2_center'].set(value=640, min=610, max=690)
    pars['g2_sigma'].set(value=20,min=10,max=35)
    pars['g2_amplitude'].set(value=0.005*20*2.5, min=0.0005)
    
    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())

    
    pars['g3_center'].set(value=750,min=700,max=780)
    pars['g3_sigma'].set(value=20,min=15,max=35)
    pars['g3_amplitude'].set(value=15*2.5*0.015, min=0.0015)
    
    gauss4 = GaussianModel(prefix='g4_')
    pars.update(gauss4.make_params())

    
    pars['g4_center'].set(value=850,min=780, max=920)
    pars['g4_sigma'].set(value=20,min=10,max=35)
    pars['g4_amplitude'].set(value=20*2.5*0.005, min=0.0005)
    
    mod = gauss1_1 + gauss1 + gauss2 + gauss3 +  gauss4 + exp_mod
    
    init = mod.eval(pars, x=Wav)
    out = mod.fit(data, pars, x=Wav)
    comps=out.eval_components(x=Wav)
    
    resonance_p1_1 = out.params['g1_1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm_p1_1 = out.params['g1_1_fwhm'].value
    height_p1_1 = out.params['g1_1_height'].value
    sigma_p1_1=out.params['g1_1_sigma'].value
    amp_p1_1=out.params['g1_1_amplitude'].value
    p1_1_fit=comps['g1_1_']
    # print(amp_p1_1)
    # print(height_p1_1)
    # print(sigma_p1_1)
    
    
    resonance_p1 = out.params['g1_center'].value
    #stderr_p1 = out.params['g1_center'].stderr
    fwhm_p1 = out.params['g1_fwhm'].value
    height_p1 = out.params['g1_height'].value
    sigma_p1=out.params['g1_sigma'].value
    amp_p1=out.params['g1_amplitude'].value
    p1_fit=comps['g1_']
    # print(amp_p1)
    # print(height_p1)
    # print(sigma_p1)
    
    resonance_p2 = out.params['g2_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p2 = out.params['g2_fwhm'].value
    height_p2 = out.params['g2_height'].value
    sigma_p2=out.params['g2_sigma'].value
    R_p2=out.params
    p2_fit=comps['g2_']
    
    resonance_p3 = out.params['g3_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p3 = out.params['g3_fwhm'].value
    height_p3 = out.params['g3_height'].value
    sigma_p3=out.params['g3_sigma'].value
    R_p3=out.params
    p3_fit=comps['g3_']
    
    resonance_p4 = out.params['g4_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p4 = out.params['g4_fwhm'].value
    height_p4 = out.params['g4_height'].value
    sigma_p4=out.params['g4_sigma'].value
    R_p4=out.params
    p4_fit=comps['g4_']
    
    bgd=comps['exp_']
    #print(out.eval_uncertainty())
    #out.plot_fit()
    #out.plot_residuals()
    #print(out.success)
    #print(out.errorbars)
    if check==True:
        pf.plt_plot(Wav,data*100,linewidth=3)
        plt.plot(Wav,p1_1_fit*100,label='T mode',linewidth=2)
        plt.plot(Wav,p1_fit*100,label='(40) mode',linewidth=2)
        plt.plot(Wav,p2_fit*100,label='(30) mode',linewidth=2)
        plt.plot(Wav,p3_fit*100,label='(20) mode',linewidth=2)
        plt.plot(Wav,p4_fit*100,label='(10) mode',linewidth=2)
        plt.plot(Wav,out.best_fit*100,label='Fitted curve',linewidth=2)
        #plt.plot(Wav,bgd)
        #print(out.chisqr)
    if out.success == False:
        raise Exception ('Fitting fails!!!')
    #print(out.fit_report(sort_pars=True))        
    u_c = out.eval_uncertainty()

    Peak_1_1=np.array([resonance_p1_1, height_p1_1, fwhm_p1_1])
    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    Peak_3=np.array([resonance_p3, height_p3, fwhm_p3])
    Peak_4=np.array([resonance_p4, height_p4, fwhm_p4])

    return Peak_1_1, Peak_1, Peak_2, Peak_3, Peak_4, out.best_fit, bgd

def Multi_Peaks_fit_Gau_3P(data, Wav,check=False,Pos=[520,560,800]):

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x=Wav)
       
    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    index_1=np.argmin(abs(Wav-500))
    index_2=np.argmin(abs(Wav-550))
    
    data0=reduceNoise(data)
    max_1=np.argmax(data0[index_1:index_2])+index_1
    pars['g1_center'].set(value=Wav[max_1], min=Pos[0]-20, max=Pos[0]+30)
    pars['g1_sigma'].set(value=24.8,min=20,max=30)
    pars['g1_amplitude'].set(value=24.8*2.5*data0[max_1], min=24.8*2.5*data0[max_1]*0.9,max=24.8*2.5*data0[max_1]*1.1)
    pars['g1_height'].set(value=data0[max_1], min=1.1*data0[max_1])

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    pars['g2_center'].set(value=Wav[np.argmax(data[index_1:index_2])+index_1+100], min=Pos[1]-10, max=Pos[1]+40)
    pars['g2_sigma'].set(value=23,min=15)
    pars['g2_amplitude'].set(min=data0[max_1]*0.02)
    
    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())

    index_3=np.argmin(abs(Wav-(Pos[2]-100)))
    index_4=np.argmin(abs(Wav-(Pos[2]+100)))
    
    pars['g3_center'].set(value=Wav[np.argmax(data0[index_3:index_4])+index_3], min=Pos[2]-100, max=Pos[2]+100)
    pars['g3_sigma'].set(value=40,min=15)
    #pars['g3_amplitude'].set(value=np.max(data[index_3:index_4]), min=0.01)
    
    mod = gauss1 + gauss2 + gauss3
    
    init = mod.eval(pars, x=Wav)
    out = mod.fit(data, pars, x=Wav)
    comps=out.eval_components(x=Wav)
    
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
    R_p2=out.params
    p2_fit=comps['g2_']
    
        
    resonance_p3 = out.params['g3_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p3 = out.params['g3_fwhm'].value
    height_p3 = out.params['g3_height'].value
    sigma_p3=out.params['g3_sigma'].value
    R_p3=out.params
    p3_fit=comps['g3_']
    if check==True:
        plt.plot(Wav, data)
        plt.plot(Wav, p1_fit)
        plt.plot(Wav, p2_fit)
        plt.plot(Wav, p3_fit)
    
    #print(out.fit_report(min_correl=0.25))        

    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    Peak_3=np.array([resonance_p3, height_p3, fwhm_p3])

    return Peak_1, Peak_2, Peak_3, out.best_fit

def Multi_Peaks_fit_Gau_3P_v2(data, Wav):

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x=Wav)
       
    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    index_1=np.argmin(abs(Wav-500))
    index_2=np.argmin(abs(Wav-550))

    data0=reduceNoise(data)
    max_1=np.argmax(data0[index_1:index_2])+index_1
    pars['g1_center'].set(value=Wav[max_1], min=500, max=550)
    pars['g1_sigma'].set(value=24.8,min=20,max=30)
    pars['g1_amplitude'].set(value=24.8*2.5*data0[max_1], min=24.8*2.5*data0[max_1]*0.9,max=24.8*2.5*data0[max_1]*1.1)
    pars['g1_height'].set(value=data0[max_1], min=1.2*data0[max_1])

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    pars['g2_center'].set(value=Wav[np.argmax(data[index_1:index_2])+index_1+100], min=550, max=700)
    pars['g2_sigma'].set(value=23,min=15,max=30)
    pars['g2_amplitude'].set(min=data0[max_1]*0.02)
    
    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())

    index_3=np.argmin(abs(Wav-700))
    index_4=np.argmin(abs(Wav-900))
    
    pars['g3_center'].set(value=Wav[np.argmax(data[index_3:index_4])+index_3], min=700, max=900)
    pars['g3_sigma'].set(value=40,min=15)
    #pars['g3_amplitude'].set(value=np.max(data[index_3:index_4]), min=0.01)
    
    
    mod = gauss1 + gauss2 + gauss3
    
    init = mod.eval(pars, x=Wav)
    out = mod.fit(data, pars, x=Wav)
    comps=out.eval_components(x=Wav)
    
    
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
    R_p2=out.params
    p2_fit=comps['g2_']
    
        
    resonance_p3 = out.params['g3_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p3 = out.params['g3_fwhm'].value
    height_p3 = out.params['g3_height'].value
    sigma_p3=out.params['g3_sigma'].value
    R_p3=out.params
    p3_fit=comps['g3_']
    
    
    #print(out.fit_report(min_correl=0.25))        

    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    Peak_3=np.array([resonance_p3, height_p3, fwhm_p3])

    return Peak_1, Peak_2, Peak_3, out.best_fit, p1_fit, p2_fit, p3_fit

def Multi_Peaks_fit_Gau_3P_v2_USP(data, Wav, check=False):

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x=Wav)
       
    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    index_1=np.argmin(abs(Wav-500))
    index_2=np.argmin(abs(Wav-550))

    data0=reduceNoise(data)
    max_1=np.argmax(data0[index_1:index_2])+index_1
    pars['g1_center'].set(value=Wav[max_1], min=500, max=550)
    pars['g1_sigma'].set(value=24.8,min=20,max=30)
    pars['g1_amplitude'].set(value=24.8*2.5*data0[max_1], min=24.8*2.5*data0[max_1]*0.9,max=24.8*2.5*data0[max_1]*1.1)
    pars['g1_height'].set(value=data0[max_1], min=1.2*data0[max_1])

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    pars['g2_center'].set(value=Wav[np.argmax(data[index_1:index_2])+index_1+100], min=550, max=700)
    pars['g2_sigma'].set(value=23,min=15,max=30)
    pars['g2_amplitude'].set(min=data0[max_1]*0.02)
    
    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())

    index_3=np.argmin(abs(Wav-700))
    index_4=np.argmin(abs(Wav-850))
    
    pars['g3_center'].set(value=730, min=700, max=830)
    pars['g3_sigma'].set(value=15,min=10)
    pars['g3_amplitude'].set(min=0.00005,max=10)

    gauss4 = GaussianModel(prefix='g4_')
    pars.update(gauss4.make_params())
    pars['g4_center'].set(value=810, min=800, max=900)
    pars['g4_sigma'].set(value=25,min=15,max=30)
    pars['g3_amplitude'].set(min=0.00005,max=10)
    
    
    mod = gauss1 + gauss2 + gauss3 + gauss4
    
    init = mod.eval(pars, x=Wav)
    out = mod.fit(data, pars, x=Wav)
    comps=out.eval_components(x=Wav)
    
    
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
    R_p2=out.params
    p2_fit=comps['g2_']
    
        
    resonance_p3 = out.params['g3_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p3 = out.params['g3_fwhm'].value
    height_p3 = out.params['g3_height'].value
    sigma_p3=out.params['g3_sigma'].value
    R_p3=out.params
    p3_fit=comps['g3_']

    # resonance_exp = out.params['exp_center'].value
    # #stderr_p2 = out.params['g2_center'].stderr
    # fwhm_exp = out.params['exp_fwhm'].value
    # height_exp = out.params['exp_height'].value
    # sigma_exp=out.params['exp_sigma'].value
    # R_exp=out.params
    # exp_fit=comps['exp_']

    resonance_p4 = out.params['g4_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p4 = out.params['g4_fwhm'].value
    height_p4 = out.params['g4_height'].value
    sigma_p4=out.params['g4_sigma'].value
    R_p4=out.params
    p4_fit=comps['g4_']
    
    #print(out.fit_report(min_correl=0.25))        

    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    Peak_3=np.array([resonance_p3, height_p3, fwhm_p3])
    Peak_4=np.array([resonance_p4, height_p4, fwhm_p4])
    
    if check==True:
        pf.plt_plot(Wav,data*100,linewidth=3)
        plt.plot(Wav,p1_fit*100,label='T mode',linewidth=2)
        plt.plot(Wav,p2_fit*100,label='(20) mode',linewidth=2)
        plt.plot(Wav,p3_fit*100,label='(10) mode',linewidth=2)
        plt.plot(Wav,p4_fit*100,label='(11) mode',linewidth=2)
        plt.plot(Wav,out.best_fit*100,label='Fitted curve',linewidth=2)
        plt.ylabel('Scattering intensity (%)')

    return Peak_1, Peak_2, Peak_3,Peak_4, out.best_fit, p1_fit, p2_fit, p3_fit,p4_fit

def Multi_Peaks_fit_Gau_3P_v2_USP_Ag(data, Wav, check=False):

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x=Wav)
       
    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    index_1=np.argmin(abs(Wav-500))
    index_2=np.argmin(abs(Wav-550))
    
    index_3=np.argmin(abs(Wav-600))
    index_4=np.argmin(abs(Wav-700))

    data0=reduceNoise(data)
    max_1=np.argmax(data0[index_1:index_2])+index_1
    pars['g1_center'].set(value=Wav[max_1], min=500, max=550)
    pars['g1_sigma'].set(value=24.8,min=20,max=40)
    pars['g1_amplitude'].set(value=24.8*2.5*data0[max_1], min=24.8*2.5*data0[max_1]*0.9,max=24.8*2.5*data0[max_1]*1.1)
    pars['g1_height'].set(value=data0[max_1], min=1.2*data0[max_1])

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    pars['g2_center'].set(value=Wav[np.argmax(data[index_1:index_2])+index_1+120], min=590, max=700)
    pars['g2_sigma'].set(value=23,min=15,max=33)
    pars['g2_amplitude'].set(min=data0[max_1]*0.02)
    
    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())

    index_3=np.argmin(abs(Wav-700))
    index_4=np.argmin(abs(Wav-850))
    
    pars['g3_center'].set(value=750, min=740, max=810)
    pars['g3_sigma'].set(value=15,min=10,max=30)
    pars['g3_amplitude'].set(min=0.05,max=10)

    gauss4 = GaussianModel(prefix='g4_')
    pars.update(gauss4.make_params())
    pars['g4_center'].set(value=850, min=800, max=950)
    pars['g4_sigma'].set(value=25,min=15,max=30)
    pars['g4_amplitude'].set(min=0.00005,max=10)
    
    
    mod = gauss1 + gauss2 + gauss3 + gauss4
    
    init = mod.eval(pars, x=Wav)
    out = mod.fit(data, pars, x=Wav)
    comps=out.eval_components(x=Wav)
    
    
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
    R_p2=out.params
    p2_fit=comps['g2_']
    
        
    resonance_p3 = out.params['g3_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p3 = out.params['g3_fwhm'].value
    height_p3 = out.params['g3_height'].value
    sigma_p3=out.params['g3_sigma'].value
    R_p3=out.params
    p3_fit=comps['g3_']

    # resonance_exp = out.params['exp_center'].value
    # #stderr_p2 = out.params['g2_center'].stderr
    # fwhm_exp = out.params['exp_fwhm'].value
    # height_exp = out.params['exp_height'].value
    # sigma_exp=out.params['exp_sigma'].value
    # R_exp=out.params
    # exp_fit=comps['exp_']

    resonance_p4 = out.params['g4_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p4 = out.params['g4_fwhm'].value
    height_p4 = out.params['g4_height'].value
    sigma_p4=out.params['g4_sigma'].value
    R_p4=out.params
    p4_fit=comps['g4_']
    
    #print(out.fit_report(min_correl=0.25))        

    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    Peak_3=np.array([resonance_p3, height_p3, fwhm_p3])
    Peak_4=np.array([resonance_p4, height_p4, fwhm_p4])
    
    if check==True:
        pf.plt_plot(Wav,data*100,linewidth=3)
        plt.plot(Wav,p1_fit*100,label='T mode',linewidth=2)
        plt.plot(Wav,p2_fit*100,label='(20) mode',linewidth=2)
        plt.plot(Wav,p3_fit*100,label='(10) mode',linewidth=2)
        plt.plot(Wav,p4_fit*100,label='(11) mode',linewidth=2)
        plt.plot(Wav,out.best_fit*100,label='Fitted curve',linewidth=2)
        plt.ylabel('Scattering intensity (%)')

    return Peak_1, Peak_2, Peak_3,Peak_4, out.best_fit, p1_fit, p2_fit, p3_fit,p4_fit

def Multi_Peaks_fit_Gau_3P_v2_USP_Pd(data, Wav,check=True):

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x=Wav)
       
    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    index_1=np.argmin(abs(Wav-500))
    index_2=np.argmin(abs(Wav-550))

    data0=reduceNoise(data)
    max_1=np.argmax(data0[index_1:index_2])+index_1
    pars['g1_center'].set(value=Wav[max_1], min=500, max=550)
    pars['g1_sigma'].set(value=20,min=10,max=25)
    pars['g1_amplitude'].set(value=16*2.5*data0[max_1], min=16*2.5*data0[max_1]*0.9,max=16*2.5*data0[max_1]*1.1)
    pars['g1_height'].set(value=data0[max_1], min=1.2*data0[max_1])

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    pars['g2_center'].set(value=Wav[np.argmax(data[index_1:index_2])+index_1+50], min=560, max=620)
    pars['g2_sigma'].set(value=23,min=15,max=25)
    pars['g2_amplitude'].set(min=data0[max_1]*0.02)
    
    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())

    index_3=np.argmin(abs(Wav-700))
    index_4=np.argmin(abs(Wav-850))
    
    pars['g3_center'].set(value=710, min=680, max=780)
    pars['g3_sigma'].set(value=15,min=10)
    pars['g3_amplitude'].set(min=0.00005,max=10)

    gauss4 = GaussianModel(prefix='g4_')
    pars.update(gauss4.make_params())
    pars['g4_center'].set(value=780, min=750, max=850)
    pars['g4_sigma'].set(value=25,min=15,max=30)
    # pars['g4_amplitude'].set(min=0.00005,max=10)
    
    
    mod = gauss1 + gauss2 + gauss3 + gauss4
    
    init = mod.eval(pars, x=Wav)
    out = mod.fit(data, pars, x=Wav)
    comps=out.eval_components(x=Wav)
    
    
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
    R_p2=out.params
    p2_fit=comps['g2_']
    
        
    resonance_p3 = out.params['g3_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p3 = out.params['g3_fwhm'].value
    height_p3 = out.params['g3_height'].value
    sigma_p3=out.params['g3_sigma'].value
    R_p3=out.params
    p3_fit=comps['g3_']

    # resonance_exp = out.params['exp_center'].value
    # #stderr_p2 = out.params['g2_center'].stderr
    # fwhm_exp = out.params['exp_fwhm'].value
    # height_exp = out.params['exp_height'].value
    # sigma_exp=out.params['exp_sigma'].value
    # R_exp=out.params
    # exp_fit=comps['exp_']

    resonance_p4 = out.params['g4_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p4 = out.params['g4_fwhm'].value
    height_p4 = out.params['g4_height'].value
    sigma_p4=out.params['g4_sigma'].value
    R_p4=out.params
    p4_fit=comps['g4_']
    
    #print(out.fit_report(min_correl=0.25))        
    if check==True:
        pf.plt_plot(Wav,data*100,linewidth=3)
        plt.plot(Wav,p1_fit*100,label='T mode',linewidth=2)
        plt.plot(Wav,p2_fit*100,label='(20) mode',linewidth=2)
        plt.plot(Wav,p3_fit*100,label='(10) mode',linewidth=2)
        plt.plot(Wav,p4_fit*100,label='(11) mode',linewidth=2)
        plt.plot(Wav,out.best_fit*100,label='Fitted curve',linewidth=2)
        plt.ylabel('Scattering intensity (%)')

    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    Peak_3=np.array([resonance_p3, height_p3, fwhm_p3])
    Peak_4=np.array([resonance_p4, height_p4, fwhm_p4])

    return Peak_1, Peak_2, Peak_3,Peak_4, out.best_fit, p1_fit, p2_fit, p3_fit,p4_fit

def Multi_Peaks_fit_Gau_3P_v2_USP_BPT(data, Wav,g1_sig=22,check=False):

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x=Wav)
       
    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    index_1=np.argmin(abs(Wav-500))
    index_2=np.argmin(abs(Wav-550))

    data0=reduceNoise(data)
    max_1=np.argmax(data0[index_1:index_2])+index_1
    pars['g1_center'].set(value=Wav[max_1], min=500, max=550)
    pars['g1_sigma'].set(value=g1_sig,min=10,max=g1_sig+3)
    pars['g1_amplitude'].set(value=g1_sig*2.5*data0[max_1], min=g1_sig*2.5*data0[max_1]*0.9,max=g1_sig*2.5*data0[max_1]*1.1)
    pars['g1_height'].set(value=data0[max_1], min=1.2*data0[max_1])

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    pars['g2_center'].set(value=Wav[np.argmax(data[index_1:index_2])+index_1+60], min=550, max=620)
    pars['g2_sigma'].set(value=23,min=15,max=25)
    pars['g2_amplitude'].set(min=data0[max_1]*0.02)
    
    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())

    index_3=np.argmin(abs(Wav-700))
    index_4=np.argmin(abs(Wav-850))
    
    pars['g3_center'].set(value=710, min=680, max=780)
    pars['g3_sigma'].set(value=15,min=10)
    pars['g3_amplitude'].set(min=0.00005,max=10)

    gauss4 = GaussianModel(prefix='g4_')
    pars.update(gauss4.make_params())
    pars['g4_center'].set(value=780, min=760, max=800)
    pars['g4_sigma'].set(value=25,min=15,max=35)
    # pars['g4_amplitude'].set(min=0.00005,max=10)
    
    
    mod = gauss1 + gauss2 + gauss3 + gauss4
    
    init = mod.eval(pars, x=Wav)
    out = mod.fit(data, pars, x=Wav)
    comps=out.eval_components(x=Wav)
    
    
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
    R_p2=out.params
    p2_fit=comps['g2_']
    
        
    resonance_p3 = out.params['g3_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p3 = out.params['g3_fwhm'].value
    height_p3 = out.params['g3_height'].value
    sigma_p3=out.params['g3_sigma'].value
    R_p3=out.params
    p3_fit=comps['g3_']

    # resonance_exp = out.params['exp_center'].value
    # #stderr_p2 = out.params['g2_center'].stderr
    # fwhm_exp = out.params['exp_fwhm'].value
    # height_exp = out.params['exp_height'].value
    # sigma_exp=out.params['exp_sigma'].value
    # R_exp=out.params
    # exp_fit=comps['exp_']

    resonance_p4 = out.params['g4_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p4 = out.params['g4_fwhm'].value
    height_p4 = out.params['g4_height'].value
    sigma_p4=out.params['g4_sigma'].value
    R_p4=out.params
    p4_fit=comps['g4_']
    
    #print(out.fit_report(min_correl=0.25))        
    if check==True:
        pf.plt_plot(Wav,data*100,linewidth=3)
        plt.plot(Wav,p1_fit*100,label='T mode',linewidth=2)
        plt.plot(Wav,p2_fit*100,label='(20) mode',linewidth=2)
        plt.plot(Wav,p3_fit*100,label='(10) mode',linewidth=2)
        plt.plot(Wav,p4_fit*100,label='(11) mode',linewidth=2)
        plt.plot(Wav,out.best_fit*100,label='Fitted curve',linewidth=2)
        plt.ylabel('Scattering intensity (%)')

    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    Peak_3=np.array([resonance_p3, height_p3, fwhm_p3])
    Peak_4=np.array([resonance_p4, height_p4, fwhm_p4])

    return Peak_1, Peak_2, Peak_3,Peak_4, out.best_fit, p1_fit, p2_fit, p3_fit,p4_fit

def Multi_Peaks_fit_Gau_3P_v2_USP_BPT_Pd(data, Wav,g1_sg=16):

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(data, x=Wav)
       
    gauss1 = GaussianModel(prefix='g1_')
    pars.update(gauss1.make_params())

    index_1=np.argmin(abs(Wav-500))
    index_2=np.argmin(abs(Wav-550))

    data0=reduceNoise(data)
    max_1=np.argmax(data0[index_1:index_2])+index_1
    pars['g1_center'].set(value=Wav[max_1], min=500, max=550)
    pars['g1_sigma'].set(value=g1_sg,min=10,max=g1_sg+2)
    pars['g1_amplitude'].set(value=g1_sg*2.5*data0[max_1], min=g1_sg*2.5*data0[max_1]*0.9,max=g1_sg*2.5*data0[max_1]*1.1)
    pars['g1_height'].set(value=data0[max_1], min=1.2*data0[max_1])

    gauss2 = GaussianModel(prefix='g2_')
    pars.update(gauss2.make_params())
    
    pars['g2_center'].set(value=Wav[np.argmax(data[index_1:index_2])+index_1+50], min=550, max=620)
    pars['g2_sigma'].set(value=23,min=15,max=25)
    pars['g2_amplitude'].set(min=data0[max_1]*0.02)
    
    gauss3 = GaussianModel(prefix='g3_')
    pars.update(gauss3.make_params())

    index_3=np.argmin(abs(Wav-700))
    index_4=np.argmin(abs(Wav-850))
    
    pars['g3_center'].set(value=710, min=680, max=780)
    pars['g3_sigma'].set(value=15,min=10)
    pars['g3_amplitude'].set(min=0.00005,max=10)

    gauss4 = GaussianModel(prefix='g4_')
    pars.update(gauss4.make_params())
    pars['g4_center'].set(value=780, min=760, max=850)
    pars['g4_sigma'].set(value=25,min=15,max=30)
    # pars['g4_amplitude'].set(min=0.00005,max=10)
    
    
    mod = gauss1 + gauss2 + gauss3 + gauss4
    
    init = mod.eval(pars, x=Wav)
    out = mod.fit(data, pars, x=Wav)
    comps=out.eval_components(x=Wav)
    
    
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
    R_p2=out.params
    p2_fit=comps['g2_']
    
        
    resonance_p3 = out.params['g3_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p3 = out.params['g3_fwhm'].value
    height_p3 = out.params['g3_height'].value
    sigma_p3=out.params['g3_sigma'].value
    R_p3=out.params
    p3_fit=comps['g3_']

    # resonance_exp = out.params['exp_center'].value
    # #stderr_p2 = out.params['g2_center'].stderr
    # fwhm_exp = out.params['exp_fwhm'].value
    # height_exp = out.params['exp_height'].value
    # sigma_exp=out.params['exp_sigma'].value
    # R_exp=out.params
    # exp_fit=comps['exp_']

    resonance_p4 = out.params['g4_center'].value
    #stderr_p2 = out.params['g2_center'].stderr
    fwhm_p4 = out.params['g4_fwhm'].value
    height_p4 = out.params['g4_height'].value
    sigma_p4=out.params['g4_sigma'].value
    R_p4=out.params
    p4_fit=comps['g4_']
    
    #print(out.fit_report(min_correl=0.25))        

    Peak_1=np.array([resonance_p1, height_p1, fwhm_p1])
    Peak_2=np.array([resonance_p2, height_p2, fwhm_p2])
    Peak_3=np.array([resonance_p3, height_p3, fwhm_p3])
    Peak_4=np.array([resonance_p4, height_p4, fwhm_p4])

    return Peak_1, Peak_2, Peak_3,Peak_4, out.best_fit, p1_fit, p2_fit, p3_fit,p4_fit



def check(Wav, data, order=0, bgd_remove=False,g1_sig=22):
    if bgd_remove:
        data=IP.Run(Wav,data,Poly_Order=order)
    a,b,c,d,e,f,g,h,i=Multi_Peaks_fit_Gau_3P_v2_USP_BPT(data,Wav,g1_sig=g1_sig)
    print(b)
    plt.plot(Wav,data*100)
    plt.plot(Wav,f*100)
    plt.plot(Wav,g*100)
    plt.plot(Wav,e*100)
    plt.plot(Wav,h*100)
    plt.plot(Wav,i*100)



def check_spe_shape(p1,p2,p3):
    if p1[0] > p2[0] or p2[0]>p3[0] or p1[1] > p2[1] or p1[1]>p3[1]:
        result=False
    else:
        result=True
    return result

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
        print('Werid point!')

    return spe_0

def bgd_r_v2 (x,y):
    i0=np.argmin(abs(x-550))
    i1=np.argmin(abs(x-650))
    i_cut=np.argmin(y[i0:i1])+i0
    spe_0=IP.Run(x[:i_cut+100],y[:i_cut+100],Poly_Order=1,Maximum_Iterations=10)
    spe_1=IP.Run(x[i_cut:],y[i_cut:],Poly_Order=1,Maximum_Iterations=10)

    if np.min(spe_1[:50])<0 :
        spe_1=spe_1-np.min(spe_1[:50])

    if np.min(spe_0[-150:-100])<0 :
        spe_0=spe_0-np.min(spe_0[-150:-100])

    spe=np.concatenate((spe_0[:-100],spe_1),axis=0)
    return spe



def Fitting_result(Peak_1,Peak_2,Peak_3,Spe,Fail_spe,Data_name,save_data=True):
    Pos_1=[]
    Int_1=[]
    Width_1=[]
    for i in Peak_1:
        Pos_1.append(i[0])
        Int_1.append(i[1])
        Width_1.append(i[2])
    Pos_1=np.asarray(Pos_1)
    Int_1=np.asarray(Int_1)
    Width_1=np.asarray(Width_1)

    Pos_2=[]
    Int_2=[]
    Width_2=[]
    for i in Peak_2:
        Pos_2.append(i[0])
        Int_2.append(i[1])
        Width_2.append(i[2])
    Pos_2=np.asarray(Pos_2)
    Int_2=np.asarray(Int_2)
    Width_2=np.asarray(Width_2)

    Pos_3=[]
    Int_3=[]
    Width_3=[]
    for i in Peak_3:
        Pos_3.append(i[0])
        Int_3.append(i[1])
        Width_3.append(i[2])
    Pos_3=np.asarray(Pos_3)
    Int_3=np.asarray(Int_3)
    Width_3=np.asarray(Width_3)

    if save_data:
        np.savez(Data_name+'_POS_INT_WIDTH_transverse_mode',Pos=Pos_1,Int=Int_1,Wid=Width_1)
        np.savez(Data_name+'_POS_INT_WIDTH_(20) mode',Pos=Pos_2,Int=Int_2,Wid=Width_2)
        np.savez(Data_name+'_POS_INT_WIDTH_coupling mode',Pos=Pos_3,Int=Int_3,Wid=Width_3)
        np.savetxt(Data_name+'_Useful_para.txt',['Total_num_spe: ' + str(len(Spe)), 'Succeeded_num_spe: ' + str(len(Peak_2)), 'Failed_num_spe: ' + str(len(Fail_spe))],fmt='%s')
    return Pos_3, Pos_2, Pos_1

def plot_hist_spe(Data_name, Wav, spectra, pos, bin_width=10, bin_num=30, box_lw=2.5,labelsize=20, x_lim=[450,950],y_lim=[], save_data=True, bgd_remove=True, order=1,start=True,end=False, normalize= False,smooth=True,return_MF=False):

    print('Start plotting fig...')
    bins=int((np.max(pos)-np.min(pos))/ bin_width)
    Pos_3=pos
    counts_pos, bins_pos = np.histogram(Pos_3,bins=bins)
    figsize=10,7
    fig, ax = plt.subplots(figsize=figsize)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(box_lw)

    dark_jet = cmap_map(lambda x: x*0.8, matplotlib.cm.jet)
    n, b, patches = plt.hist(bins_pos[:-1], bins=bins_pos, weights=counts_pos,color='grey', ec='black',linewidth=1,alpha=1)

    b_centers = 0.5 * (b[:-1] + b[1:])
    b_c_new=[]
    patches_new=[]
    for i in range(len(counts_pos)):
        if counts_pos[i] >= bin_num:
            b_c_new.append(b_centers[i])
            patches_new.append(patches[i])

    b_c_new=np.asarray(b_c_new)
    b_centers=b_c_new
    col = b_centers - min(b_centers)
    col=col/max(col)

    col_n=np.asarray(range(len(col)+2))
    col_n=col_n/max(col_n)

    for c, p in zip(col_n[2:], patches_new):
        plt.setp(p, 'facecolor', dark_jet(c),alpha=0.7)
    resonance_pos, fwhm_pos, best_fit_pos, best_fit_x_pos=Gaussian(counts_pos,bins_pos[:-1]+0.5*(bins_pos[1]-bins_pos[0]))
    most_bins_pos_index=np.argmin(abs(resonance_pos-(bins_pos+0.5*(bins_pos[1]-bins_pos[0]))))
    plt.plot(best_fit_x_pos,best_fit_pos,'k--',linewidth=3)
    plt.tick_params(direction='in', length=8, width=3, labelcolor='black', colors='black', which='both',top=False,left=False)
    plt.xlim(x_lim[0],x_lim[1])
    # plt.yticks([0,50,100,150])
    plt.xlabel('Wavelength (nm)', multialignment='center', rotation=0, fontsize=labelsize)
    plt.ylabel('Number', multialignment='center', rotation=-90, fontsize=labelsize,labelpad=25)
    plt.tick_params(labelsize=labelsize,pad=9)

    index_0=np.argmin(abs(Wav-500))
    index_1=np.argmin(abs(Wav-620))
    index_2=np.argmin(abs(Wav-900))
    index_3=np.argmin(abs(Wav-470))
    index_4=np.argmin(abs(Wav-530))
    ax2=ax.twinx()
    ax2_y_max=[]
    ax2_y_min=[]
    bins_spe=[]
    bins_spe_1=[]
    bins_pos_1=[]
    bins_spe_all=[]
    index_all=[]
    
    for i in range(len(bins_pos[:-1])):
        aver_spe_b=[]
        width_b=bins_pos[1]-bins_pos[0]
        index=[]
        for j in range(len(Pos_3)):
            if bins_pos[i]<= Pos_3[j] < bins_pos[i+1]:
                aver_spe_b.append(spectra[j])
                index.append(j)
        bins_spe_all.append(aver_spe_b)
        index_all.append(index)
        if len(aver_spe_b)>=bin_num:
            bins_spe_1.append(aver_spe_b)
            bins_pos_1.append(bins_pos[i])
            aver_spe_b=np.mean(aver_spe_b,axis=0)
            if smooth:
                aver_spe_b=reduceNoise(aver_spe_b, cutoff = 1000, fs = 80000, factor = 7)
            if bgd_remove:
                aver_spe_b=IP.Run(Wav,aver_spe_b,Poly_Order=order)
                if start:
                    aver_spe_b=aver_spe_b-np.min(aver_spe_b[index_3:index_4])
                elif end:
                    aver_spe_b=aver_spe_b-np.min(aver_spe_b[-50:])

            if normalize:
                aver_spe_b= aver_spe_b/(np.max(aver_spe_b[index_0:index_1]))
            ax2_y_max.append(np.max(aver_spe_b[index_0:index_2])) # for setting the range of y_axis
            ax2_y_min.append(np.min(aver_spe_b[:index_2]))
            #plt.plot(aver_spe_b)
            #print(ax2_y_max)
            bins_spe.append(aver_spe_b)
    bins_spe=np.asarray(bins_spe)

    for i in range(len(bins_spe)):
        if normalize:
            ax2.plot(Wav,bins_spe[i],color=matplotlib.cm.jet(col_n[i+2]))
        else:
            ax2.plot(Wav,bins_spe[i]*100,color=matplotlib.cm.jet(col_n[i+2]))

    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")
    if normalize:
        ax2.set_ylabel('Scattering intensity', multialignment='center', rotation=90, fontsize=labelsize)
    else:
        ax2.set_ylabel('Scattering intensity (%)', multialignment='center', rotation=90, fontsize=labelsize)
    if normalize and y_lim == []: 
        ax2.set_ylim(np.min(ax2_y_min)-abs(np.min(ax2_y_min)*0.2),np.max(ax2_y_max)*1.2)
    elif y_lim == []:
        ax2.set_ylim(np.min(ax2_y_min)*100-abs(np.min(ax2_y_min)*100*0.2),np.max(ax2_y_max)*1.2*100)
    else:
        ax2.set_ylim(y_lim[0],y_lim[1])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.tick_params(labelsize=labelsize,pad=9)
    plt.tick_params(direction='in', length=8, width=3, labelcolor='black', colors='black', which='both',top=False,right=False)
    plt.subplots_adjust(top=0.975,bottom=0.115,left=0.155,right=0.875,hspace=0.2,wspace=0.2)
    if save_data:
        np.savetxt(Data_name+'_pos_3_hist.txt',['resonance_pos: ' +str(resonance_pos), 'fwhm_pos: '+str(fwhm_pos)],fmt='%s')       
        np.savez(Data_name+'_All_spe',c_pos=pos,spectra=spectra,Wav=Wav)
        np.savez(Data_name+'_bin_center_spectra_wid',Wav=Wav,spectra=bins_spe,bins=bins)
        plt.savefig(Data_name+'_all_hist_bins_wid.jpg',dpi=300)
    if return_MF:
        return Wav, bins_spe, bins_spe_1, bins_pos_1, index_all




