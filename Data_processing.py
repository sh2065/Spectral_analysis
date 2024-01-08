# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:09:42 2021

@author: sh2065
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from Shu_analysis import plt_fig as pf
from Shu_analysis import DF_sorting as DFS
from scipy import interpolate
from scipy.signal import find_peaks
from nplab.analysis.background_removal import Adaptive_Polynomial as AP
from tqdm import tqdm
from lmfit.models import GaussianModel, LorentzianModel
from Shu_analysis import Fitting_functions as Ff
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter as sgFilt

def UV_vis(File_path, data_path, input_bgd_ref =False, bgd_input= None, ref_input=None,plot=False):
    
    with h5py.File(File_path,'r') as File:
        spe=np.asarray(File[data_path])
        Wav = np.asarray(File[data_path].attrs['wavelengths'])
        bgd=np.asarray(File[data_path].attrs['background'])
        ref=np.asarray(File[data_path].attrs['reference'])

    if input_bgd_ref:
        bgd = bgd_input
        ref = ref_input
    A=np.log10((ref-bgd)/(abs(spe-bgd)))
    if plot:
        pf.plt_plot(Wav, A,linewidth=2)

    return Wav, A


def sort_h5py_bug(File_path):
    with h5py.File(File_path,'r+') as File:
        for i in File:
            time_0= str(File[i].attrs['creation_timestamp'])
            del File[i].attrs['creation_timestamp']
            File[i].attrs['creation_timestamp_new'] = bytes(time_0, encoding='utf-8')
            if type(File[i]) is h5py._hl.group.Group:
                for j in File[i]:
                    time_1= File[i][j].attrs['creation_timestamp']
                    del File[i][j].attrs['creation_timestamp']
                    File[i][j].attrs['creation_timestamp_new'] = bytes(time_1, encoding='utf-8')
                    if type(File[i][j]) is h5py._hl.group.Group:
                        for k in File[i][j]:
                            time_2= File[i][j][k].attrs['creation_timestamp']
                            del File[i][j][k].attrs['creation_timestamp'] 
                            File[i][j][k].attrs['creation_timestamp_new'] = bytes(time_2, encoding='utf-8')
                            if type(File[i][j][k]) is h5py._hl.group.Group:
                                for h in File[i][j][k]:
                                    time_3= File[i][j][k][h].attrs['creation_timestamp']
                                    del File[i][j][k][h].attrs['creation_timestamp']
                                    File[i][j][k][h].attrs['creation_timestamp_new'] = bytes(time_3, encoding='utf-8')


def copy_h5file( file_source, file_dest):
    file_0= h5py.File(file_dest, 'w')
    file_1= h5py.File(file_source, 'r')
    group = list(file_1.keys())
    for i in group:
        # file_0.create_group(i)
        file_1.copy(file_1[i],file_0)
    file_0.close()
    file_1.close()

def SERS_spe_sorting(File_path, scan=[0], group_len=2, data_name='kinetic_SERS', ifs=False):
    spe_all=[]
    Int_all=[]
    with h5py.File(File_path, 'r') as File:
        for k in scan:
            data=File['/ParticleScannerScan_'+str(k)]
            for i,j in enumerate(data):
                if j[0] == 'P' and len(data[j]) >= group_len:
                    Wav=data[j][data_name].attrs['wavelengths']
                    if Wav[1] < Wav[0]:
                        Wav=np.flipud(Wav)
                        spe_0=np.fliplr(data[j][data_name])
                    else:
                        spe_0=data[j][data_name]
                    if ifs:
                        spe_0=np.average(spe_0[1:],axis=0)
                    else:
                        spe_0=np.average(spe_0,axis=0)
                    spe_all.append(spe_0)
                    Int_all.append(np.max(spe_0))
    return Wav, spe_all , Int_all

def DF_spe_sorting(File_path,data_path, x_lim=[480,950], z_scan=True):
    
    with h5py.File(File_path, 'r') as File:
        data=np.asarray(File[data_path])
        bgd=File[data_path].attrs['background']
        ref=File[data_path].attrs['reference']
        Wav=File[data_path].attrs['wavelengths']
    if len( data.shape) == 2 and z_scan:
        data=np.max(data,axis=0)
    else:
        pass

    spe=(data-bgd)/(ref-bgd)
    i_0=np.argmin(abs(Wav-x_lim[0]))
    i_1=np.argmin(abs(Wav-x_lim[1]))
    if z_scan:
        Wav, spe = Wav[i_0:i_1], spe[i_0:i_1]
    else:
        Wav, spe = Wav[i_0:i_1], spe[:,i_0:i_1]

    return Wav, spe

def DF_spe_sorting_trandor(Wav, File_path, DF_path, ref_path, bgd_ref_path, bgd_path, x_range=[550,1000],notch_region=[617,644,764,794], plot=False,compare=False):
    index_0=np.argmin(abs(x_range[0]-Wav))
    index_1=np.argmin(abs(x_range[1]-Wav))
    
    Path_combine= [DF_path,DF_path[:-1]+ '1', ref_path, bgd_ref_path, bgd_path]
    Wav_wrong, Spe, Spe_all = Output_data_random(File_path, Path_combine)

    Spe_DF=(Spe_all[0]-Spe_all[4])/(Spe_all[2]-Spe_all[3])
    Spe_laser_DF=(Spe_all[1]-Spe_all[4])/(Spe_all[2]-Spe_all[3])

    Wav=Wav[index_0:index_1]
    Spe_DF=Spe_DF[index_0:index_1]
    Spe_laser_DF=Spe_laser_DF[index_0:index_1]
    
    index_2=np.argmin(abs(notch_region[0]-Wav))
    index_3=np.argmin(abs(notch_region[1]-Wav))
    index_4=np.argmin(abs(notch_region[2]-Wav))
    index_5=np.argmin(abs(notch_region[3]-Wav))
    
    Wav_new=Wav
    Spe_DF_new=np.concatenate((Spe_DF[:index_2],np.zeros(len(Spe_DF[index_2:index_3])), Spe_DF[index_3:index_4],np.zeros(len(Spe_DF[index_4:index_5])), Spe_DF[index_5:]),axis=0)
    Spe_laser_DF_new=np.concatenate((Spe_laser_DF[:index_2],np.zeros(len(Spe_laser_DF[index_2:index_3])), Spe_laser_DF[index_3:index_4],np.zeros(len(Spe_laser_DF[index_4:index_5])), Spe_laser_DF[index_5:]),axis=0)

    if plot and compare:
        pf.plt_plot(Wav_new, Spe_DF_new)
        plt.plot(Wav_new, Spe_laser_DF_new)
    elif plot:
        pf.plt_plot(Wav_new, Spe_DF_new*100)
        plt.ylabel('Scattering intensity (%)')
    return Wav_new, Spe_DF_new


def PL_spe_sorting(File_path,data_path, DC='',power=1, exp=1,mg=1):
    with h5py.File(File_path, 'r') as File:
        spe_all=[]
        for i in data_path:
            data=np.asarray(File[i])
            if DC=='':
                bgd=File[i].attrs['background']
                DC=bgd
            Wav=File[i].attrs['wavelengths']
            spe=(data-DC)/power/exp*mg
            if len(spe.shape) == 2:
                spe=spe[1:]
                spe_all.append(np.mean(spe,axis=0))
            else:
                spe_all.append(spe)
        if len(spe_all) == 1:
            spe_all = spe
        spe_all =np.asarray(spe_all)
    return Wav, spe_all

def plt_ctf_fig(File_path,data_path, DC=0,power=1, exp=0.2,mg=1,offset=0,Raman=True, despike=False,name=''):
    with h5py.File(File_path, 'r') as File:
        data_spe=np.asarray(File[data_path])
        Wav=File[data_path].attrs['wavelengths']
    data_spe=np.reshape(data_spe,(-1,1600))
    if Wav[0] > Wav [1]:
        Wav=np.flipud(Wav)
        data_spe=np.fliplr(data_spe)
    x=Wav
    x_label='Wavelengths (nm)'
    if Raman:
        Ramanshift=(1/633-1/Wav)*10**7 +offset
        x=Ramanshift
        x_label='Ramanshift (cm$^{-1}$)'
    if despike:
        data_spe=DFS.remove_PP(data_spe)
    
    data_spe=(data_spe-DC)/exp/power/mg
    Time=exp*np.arange(len(data_spe))
    pf.plt_contourf(x, Time, data_spe,color_range=[1,0.6],pcolormesh=True)
    plt.xlabel(x_label)
    # if len(list(str(mg))) > 1:
    #     n=len(list(str(mg)))
    #     plt.ylabel('Intensity (x 10' + '$^{'+str(n)+'}$' +'cts/mW/s)')
    # else:
    #     plt.ylabel('Intensity (cts/mW/s)')
    plt.ylabel('Time (s)')
    if name != '':
        plt.savefig(name,dpi=300)
        np.savez(name,Ramanshift=x, data=data_spe,Time=Time)


def plt_figs(Wav,data, mg=1,figsize=[8,6],xlim=[], linewidth=2, box_lw =2, new_fig=True,label_s=20,color='C0',ylabel='',name='', offset=0):
    if len(data.shape) == 2:
        for i,j in enumerate(data):
            if i == 0:
                pf.plt_plot(Wav, j*mg + i*offset,linewidth=linewidth,figsize=figsize,new_fig=new_fig, box_lw =box_lw, label_s=label_s)
            else:
                plt.plot(Wav, j*mg+ i*offset,linewidth=linewidth)
    else:
        pf.plt_plot(Wav, data*mg,linewidth=linewidth,figsize=figsize,new_fig=new_fig, box_lw =box_lw, label_s=label_s,color=color)
    plt.xlabel('Wavelength (nm)')
    if ylabel=='':
        plt.ylabel('Scattering intensity (%)')
    else:
        plt.ylabel(ylabel)
    if figsize!=[8,6]:
        plt.tight_layout()
    if xlim!=[]:
        plt.xlim(xlim[0],xlim[1])
    if name != '':
        plt.savefig(name+'.jpg',dpi=300)

def Output_data(File_path, data_path, data_name='',x_cut=[405,950],mg=100,mulit=True, DF=False,PL=False, plot=False, new_fig=True, normalized=False):
    data_c=[]
    with h5py.File(File_path, 'r') as File:
        data=File[data_path]
        if mulit: # tracking data  with a lot of particles.
            for i,j in enumerate(data):
                if DF and j[0] == 'P':
                    Wav, spe = DF_spe_sorting(File,data_path +'/'+ j +'/'+ data_name)
                    data_c.append(spe)
                elif PL and j[0] == 'P':
                    spe=np.asarray(data[j][data_name])
                    bgd=data[j][data_name].attrs['background']
                    Wav=data[j][data_name].attrs['wavelengths']
                    spe=np.average((spe-bgd),axis=0)
                    data_c.append(spe)
                elif j[0] == 'P':
                    spe=np.asarray(data[j][data_name])
                    Wav=data[j][data_name].attrs['wavelengths']
                    data_c.append(spe)
                else:
                    print('object '+j + ' does not have data')
        elif DF:
            Wav, spe = DF_spe_sorting(File,data_path +'/'+ data_name)
            data_c=spe

        else:
            spe=np.asarray(data[data_name])
            Wav=data[data_name].attrs['wavelengths']
            data_c=spe
        index_0=np.argmin(abs(Wav-x_cut[0]))
        index_1=np.argmin(abs(Wav-x_cut[1]))
        data_c=np.asarray(data_c)
        if PL and x_cut[0] == 405:
            index_0=np.argmin(abs(Wav-580))
            index_1=np.argmin(abs(Wav-720))
        if PL and mg == 100:
                mg=0.000001

        if plot and mulit:
            data_c=data_c[:,index_0:index_1]
            Wav=Wav[index_0:index_1]
            plt_figs(Wav,data_c, mg,new_fig=new_fig)
        elif plot:
            data_c=data_c[index_0:index_1]
            Wav=Wav[index_0:index_1]
            plt_figs(Wav,data_c, mg, new_fig=new_fig)

    return Wav, data_c

def Output_data_random(File_path, data_path,DC=0, plot=False, laser='',offset=0):
    with h5py.File(File_path, 'r') as File:
        Spe=[]
        Spe_plot=[]
        for i in data_path:
            try:
                Wav=File[i].attrs['wavelengths']
            except:
                raise Exception('Remember to make data_path a list')
            data=np.asarray(File[i])
            shape=len(np.shape(data))

            if Wav[0] > Wav[-1]:
                Wav=np.flipud(Wav)
                if shape == 2:
                    data=np.fliplr(data)
                else:
                    data=np.flipud(data)
            if shape == 2:
                Spe_plot.append(np.mean(data,axis=0)-DC)
            else:
                Spe_plot.append(data-DC)
            Spe.append(data-DC)
        if laser!='':
            Ramanshift=(1/laser-1/Wav)*10**7+offset
            Wav=Ramanshift
        if plot:
            plt_figs(Wav,np.asarray(Spe_plot), mg=1,figsize=[8,6],new_fig=True,ylabel='Intensity',offset=offset)
        if len(Spe) == 1 and len(Spe_plot) == 1:
            Spe=Spe[0]
            Spe_plot=Spe_plot[0]
    Spe, Spe_plot = np.asarray(Spe), np.asarray(Spe_plot)
    return Wav, Spe, Spe_plot

def Data_combing(File_path, name, data_type='PL',DC=0, power=1,exposure=1, plot=False,new_fig=True):
    data_c=[]
    with h5py.File(File_path, 'r') as File:
        for i in name:
            data=File[i]
            Wav=File[i].attrs['wavelengths']
            data=np.asarray(data)
            if data_type== 'PL':
                bgd=File[i].attrs['background']
                data_c.append((data-bgd)/power/exposure)
            else:
                data_c.append((data-DC)/power/exposure)
    data_c=np.asarray(data_c)
    data_c_ave=np.mean(data_c,axis=0)
    if plot:
        plt_figs(Wav,data_c, mg=0.000001,figsize=[8,6], new_fig=new_fig)
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Intensity (x 10{^6} cts/mW/s)')
    return Wav, data_c, data_c_ave

def plt_DF_PL_bgd(File_path='',data_path_DF='',data_path_PL='',data_path_PL_bgd='',PL_exp=1,power=1,mg=1,name=''):
    with h5py.File(File_path, 'r') as File:
        Wav=File[data_path_DF].attrs['wavelengths']
        bgd=File[data_path_DF].attrs['background']
        ref=File[data_path_DF].attrs['reference']
        i_500=np.argmin(abs(Wav-500))
        i_505=np.argmin(abs(Wav-505))
        i_950=np.argmin(abs(Wav-900))
        i_900=np.argmin(abs(Wav-900))
        data_DF=np.max(File[data_path_DF],axis=0)
        data_DF=(data_DF-bgd)/(ref-bgd)*100
        DF_spe=data_DF[i_500:i_950]
        Wav_DF=Wav[i_500:i_950]
        data_PL=np.asarray(File[data_path_PL])
        data_PL_ave=((np.average(data_PL[1:],axis=0))-bgd)/PL_exp/power
        data_PL_ave=data_PL_ave[i_505:i_900]*mg
        data_PL_bgd=np.asarray(File[data_path_PL_bgd])
        data_PL_bgd=(np.average(data_PL_bgd[1:],axis=0)-bgd)/PL_exp/power
        data_PL_bgd=(data_PL_bgd[i_505:i_900])*mg
        Wav_PL=Wav[i_505:i_900]
    pf.plt_plot(Wav_DF,DF_spe,figsize=[9,6])
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Scattering intensity (%)')
    plt.twinx()
    plt.plot(Wav_PL,data_PL_ave,linewidth=3, color='C1')
    plt.plot(Wav_PL,data_PL_bgd,linewidth=3,color='C2')
    plt.ylabel('Intensity (x $10^6$ cts/mW/s)',fontsize=16)
    plt.tick_params(labelsize=16)
    plt.subplots_adjust(top=0.98,bottom=0.115,left=0.095,right=0.89,hspace=0.2,wspace=0.2)
    if name != '':
        plt.savefig(name+'.jpg',dpi=300)
    return Wav_PL, data_PL_ave, data_PL_bgd

def plt_DF_PL_trandor(File_path='',data_path_DF='',data_path_PL='',data_path_PL_bgd='',Wav_PL='',PL_exp=1,power=1,mg=1,name=''):
    with h5py.File(File_path, 'r') as File:
        Wav=File[data_path_DF].attrs['wavelengths']
        bgd=File[data_path_DF].attrs['background']
        ref=File[data_path_DF].attrs['reference']
        i_500=np.argmin(abs(Wav-450))
        i_505=np.argmin(abs(Wav-505))
        i_950=np.argmin(abs(Wav-900))
        i_900=np.argmin(abs(Wav-900))
        data_DF=np.max(File[data_path_DF],axis=0)
        data_DF=(data_DF-bgd)/(ref-bgd)*100
        DF_spe=data_DF[i_500:i_950]
        Wav_DF=Wav[i_500:i_950]
        data_PL_bgd=np.mean(File[data_path_PL_bgd],axis=0)

        data_PL=np.asarray(File[data_path_PL])
        data_PL_ave=(data_PL-data_PL_bgd)/PL_exp/power
        data_PL_ave=np.flipud(data_PL_ave)*mg

    pf.plt_plot(Wav_DF,DF_spe,figsize=[9,6])
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Scattering intensity (%)')
    plt.yticks([0,1])
    plt.twinx()
    pf.plt_plot([],[], new_fig=False)
    plt.plot(Wav_PL,data_PL_ave,linewidth=3, color='C3',zorder=-10)
    plt.ylabel('Intensity (x $10^3$ cts/mW/s)',fontsize=20)
    plt.tick_params(labelsize=20, right=True)
    plt.subplots_adjust(top=0.98,bottom=0.115,left=0.095,right=0.89,hspace=0.2,wspace=0.2)
    
    if name != '':
        plt.savefig(name+'.jpg',dpi=300)
    return Wav_PL, data_PL_ave

def plt_DF_PL(File_path='',data_path_DF='',data_path_PL='',Spe=-1,PL_exp=1,power=1,mg=1,ylim=[],name='',plot=True, PL_range=[]):
    with h5py.File(File_path, 'r') as File:
        Wav=File[data_path_DF].attrs['wavelengths']
        bgd=File[data_path_DF].attrs['background']
        ref=File[data_path_DF].attrs['reference']
        i_500=np.argmin(abs(Wav-500))
        i_950=np.argmin(abs(Wav-900))
        if PL_range == []:
            i_550=np.argmin(abs(Wav-530))
            i_750=np.argmin(abs(Wav-800))
        else:
            i_550=np.argmin(abs(Wav-PL_range[0]))
            i_750=np.argmin(abs(Wav-PL_range[1]))
        data_DF=np.max(File[data_path_DF],axis=0)
        data_DF=(data_DF-bgd)/(ref-bgd)*100
        DF_spe=data_DF[i_500:i_950]
        Wav_DF=Wav[i_500:i_950]
        data_PL=np.asarray(File[data_path_PL])
        data_PL=(data_PL[Spe]-bgd)/PL_exp/power
        data_PL=data_PL[i_550:i_750]*mg
        Wav_PL=Wav[i_550:i_750]
    if plot:
        pf.plt_plot(Wav_DF,DF_spe,figsize=[9,6],label_s=16)
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Scattering intensity (%)')
        plt.twinx()
        plt.plot(Wav_PL,data_PL,linewidth=3, color='C1')
        if ylim!=[]:
            plt.ylim(ylim[0],ylim[1])
        plt.ylabel('PL Intensity (x $10^4$ cts/mW/s)',fontsize=16)
        plt.tick_params(labelsize=16)
        plt.subplots_adjust(top=0.98,bottom=0.115,left=0.095,right=0.89,hspace=0.2,wspace=0.2)
        if name != '':
            plt.savefig(name+'.jpg',dpi=300)
    return Wav_PL, data_PL/mg,Wav_DF, DF_spe

def interpolate_fun(x, y, size=16000):
    x_new=np.linspace(np.min(x),np.max(x),size)
    fun=interpolate.interp1d(x,y,kind='quadratic')
    y_new=fun(x_new)
    return x_new, y_new

def Check_Peaks(x,y,peaks):
    plt.subplots()
    plt.plot(x, y)
    peaks_num=[np.argmin(abs(x-i)) for i in peaks]
    plt.plot(peaks, y[peaks_num], "x")

def Find_Peaks(x,y,width=3,prominence=0.01,distance=100,check=False):
    x_new, y_new=interpolate_fun(x,y)
    peaks_num,_=find_peaks((y_new-np.min(y_new))/np.max((y_new-np.min(y_new))), width=width,prominence=prominence,distance=distance)
    peaks=x_new[peaks_num]
    if check:
        Check_Peaks(x_new, y_new, peaks)
    return x_new, y_new, peaks


def cal_BPT(std_BPT_x, std_BPT_y, rf,spe,width=3,prominence=0.01,distance=100, laser='633'):

    std_BPT_x_ip, std_BPT_y_ip, standard_peaks= Find_Peaks(std_BPT_x, std_BPT_y,width=width,prominence=prominence,distance=distance)
    
    if laser =='785':
        rf_ip,spe_ip,measure_peaks_0= Find_Peaks(rf, spe, prominence=0.02)
        measure_peaks=[]
        remove_peaks=[0,  5, 14, 17, 20, 21]
        for i,j in enumerate(measure_peaks_0):
            if i not in remove_peaks:
                measure_peaks.append(j)
        measure_peaks=np.append(measure_peaks,1452.87759655)
        standard_peaks=np.append(standard_peaks,1566.74584714)
    else:
        rf_ip,spe_ip,measure_peaks= Find_Peaks(rf, spe,width=width,prominence=prominence,distance=distance)

    index=[np.argmin(abs(rf_ip-i)) for i in measure_peaks]
    rf_cal=[]
    for i,j in enumerate(index):
        if i+1 == len(index):
            step=(np.max(rf_cal)-np.min(rf_cal))/len(rf_cal)
            rf_cal_0=[standard_peaks[i]+step*k for k in range(len(rf_ip[index[i]:]))]
            rf_cal.extend(rf_cal_0)
        else:
            rf_cal_0=np.linspace(standard_peaks[i],standard_peaks[i+1],len(rf_ip[index[i]:index[i+1]]))
            rf_cal.extend(rf_cal_0)
    rf_cal=np.asarray(rf_cal)
    rf_cal_0=[standard_peaks[0]-step*h for h in range(len(rf_ip[:index[0]]))]
    rf_cal_0=np.flipud(rf_cal_0)
    rf_cal_1=np.concatenate((rf_cal_0,rf_cal),axis=0)
    
    # delete duplicates in the array
    rf_cal_2 = [i for n, i in enumerate(rf_cal_1) if i not in rf_cal_1[:n]]
    spe_ip_1= [spe_ip[n] for n, i in enumerate(rf_cal_1) if i not in rf_cal_1[:n]]
    
    # set back to the length of 1600
    rf_re_ip=np.linspace(np.min(rf_cal_2),np.max(rf_cal_2),1600)
    fun=interpolate.interp1d(rf_cal_2,spe_ip_1,kind='quadratic')
    spe_re_ip=fun(rf_re_ip)
    
    return rf_re_ip, spe_re_ip # return the calibrated Ramanshift 

def wavelengths_cal(x, standard_peaks,measure_peaks, laser=633):

    index=[np.argmin(abs(x-i)) for i in measure_peaks]
    rf_cal=[]
    for i,j in enumerate(index):
        if i+1 == len(index):
            step=(np.max(rf_cal)-np.min(rf_cal))/len(rf_cal)
            rf_cal_0=[standard_peaks[i]+step*k for k in range(len(x[index[i]:]))]
            rf_cal.extend(rf_cal_0)
        else:
            step = (standard_peaks[i+1] -standard_peaks[i])/len(x[index[i]:index[i+1]])
            rf_cal_0=np.linspace(standard_peaks[i],standard_peaks[i+1]-step,len(x[index[i]:index[i+1]]))
            rf_cal.extend(rf_cal_0)
    rf_cal=np.asarray(rf_cal)
    rf_cal_0=[standard_peaks[0]-step*(h+1) for h in range(len(x[:index[0]]))]
    rf_cal_0=np.flipud(rf_cal_0)
    rf_cal_1=np.concatenate((rf_cal_0,rf_cal),axis=0)

    if laser =='':
        return rf_cal_1 #  returning wavelength for the DF data that calibrated with Ocean optics
    else:
        Wav_cal=1/(1/laser-rf_cal_1/(10**7))
        return Wav_cal, rf_cal_1 # return the calibrated Ramanshift 

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


def td_calibration (File_path=[], data_path=[], spe_range=[510,1050], peaks_oo_remove=[627], peaks_td_remove=[], check=False):

    print('File path 1 is for Ocean optics; File path 2 is for Trandor')
    Wav_oo, data_oo, Spe_oo = Output_data_random(File_path[0], data_path[0],DC=0, plot=False, laser='',offset=0)
    Wav_td, data_td, Spe_td = Output_data_random(File_path[1], data_path[1],DC=0,plot=False, laser='',offset=0)
    
    if Wav_oo[1]< Wav_oo[0]:
        Wav_oo=np.fliplr(Wav_oo)
    if Wav_td[1]< Wav_td[0]:
        Wav_td=np.fliplr(Wav_td)
    
    if spe_range != []:
        index_0 = np.argmin(abs(Wav_oo - spe_range[0]))
        index_1 = np.argmin(abs(Wav_oo - spe_range[1]))
        Wav_oo=Wav_oo[index_0:index_1]
        Spe_oo= Spe_oo[index_0:index_1]

    Spe_td = Normalize_mean(Spe_td)
    Spe_td_sms= reduceNoise(Spe_td)
    x_td, y_td, peaks_td= Find_Peaks( Wav_td, Spe_td_sms, width=3,prominence=0.005,distance=100,check=False)
    x_oo, y_oo, peaks_oo= Find_Peaks(Wav_oo, Normalize_mean(Spe_oo), width=3,prominence=0.002,distance=100,check=False)
    
    if peaks_oo_remove != []:
        index_oo=[np.argmin(abs(peaks_oo-i)) for i in peaks_oo_remove]
        peaks_std = [j for i,j in enumerate(peaks_oo) if i not in index_oo]
    else:
        peaks_std = peaks_oo
    
    if peaks_td_remove != []:
        index_td=[np.argmin(abs(peaks_td-i)) for i in peaks_td_remove]
        peaks_measure = [j for i,j in enumerate(peaks_td) if i not in index_td]
    else:
        peaks_measure = peaks_td

    if check:
        Check_Peaks(x_td, y_td, peaks_td)
        Check_Peaks(x_oo, y_oo, peaks_std)

    Wav_cal  = wavelengths_cal(Wav_td, peaks_std, peaks_measure, laser='')
    plt.subplots()
    plt.plot(Wav_cal, Spe_td)
    plt.plot(Wav_oo, Normalize_mean(Spe_oo))
    
    res= {'Standard peaks': peaks_std, 'Measured peaks': peaks_td}

    return Wav_cal, res


def DF_SERS_correlation (Wav_DF, Spe_DF, Wav_SERS_633, Spe_SERS_633, Wav_SERS_785, Spe_SERS_785,mg_633=1,mg_785=1,label_s=22, fill=True,xlim=[600,900], colormap='Blues'):
    colors=pf.color_segmentation(color_name=colormap, steps=12)
    # colors.reverse()
    if len(Spe_DF) ==2 :
        pf.plt_plot(Wav_DF,Spe_DF[0]*100,color=colors[-6],linewidth=2,lineshape='--',figsize=[12,6],label_s=label_s, box_lw=3)
        plt.plot(Wav_DF,Spe_DF[1]*100,color='red')
    else:
        pf.plt_plot(Wav_DF,Spe_DF*100,color=colors[-6],linewidth=2,lineshape='--',figsize=[12,6],label_s=label_s, box_lw=3)
        plt.ylim(-0.05, np.max(Spe_DF*100)+0.1)
        plt.yticks(np.arange(0,np.max(Spe_DF*100)+0.1,1))
    plt.ylabel('Scattering intensity (%)')
    plt.twinx()
    plt.plot(Wav_SERS_633, Spe_SERS_633/1000*mg_633, color='C1')
    plt.plot(Wav_SERS_785, Spe_SERS_785/1000*mg_785, color='C5')
    plt.ylabel('SERS intensity (x 10$^{3}$ cts/mW/s)',fontsize=label_s)
    plt.tick_params(direction='in',labelsize=label_s,length=8,width=3,labelcolor='black', colors='black', which='both',top=False,right=True)
    plt.xlim(xlim[0],xlim[1])
    plt.subplots_adjust(top=0.98,bottom=0.14,left=0.095,right=0.93,hspace=0.2,wspace=0.2)
    if fill:
        plt.fill_between(Wav_SERS_633, Spe_SERS_633/1000*mg_633,np.zeros(shape=len(Spe_SERS_633)),alpha=0.5, color='C1')
        plt.fill_between(Wav_SERS_785, Spe_SERS_785/1000*mg_785,np.zeros(shape=len(Spe_SERS_785)),alpha=0.5, color='C5')

def DF_SERS_correlation_normal (Wav_DF, Spe_DF, Wav_SERS_633, Spe_SERS_633, Wav_SERS_785, Spe_SERS_785,mg_633=1,mg_785=1,label_s=22, fill=True,xlim=[600,900]):

    if len(Spe_DF) ==2 :
        pf.plt_plot(Wav_DF,Spe_DF[0]*100,color='black',linewidth=2,lineshape='--',figsize=[12,6],label_s=label_s, box_lw=3)
        plt.plot(Wav_DF,Spe_DF[1]*100,color='red')
    else:
        pf.plt_plot(Wav_DF,Spe_DF*100,color='black',linewidth=2,lineshape='--',figsize=[12,6],label_s=label_s, box_lw=3)
        plt.ylim(-0.01*np.max(Spe_DF*100), np.max(Spe_DF*100)+0.1)
        # plt.yticks(np.arange(0,np.max(Spe_DF*100)+0.1,1))
    plt.ylabel('Scattering intensity (%)')
    plt.twinx()
    plt.plot(Wav_SERS_633, Spe_SERS_633/10000*mg_633, color='C0')
    plt.plot(Wav_SERS_785, Spe_SERS_785/10000*mg_785, color='C1')
    plt.ylabel('SERS intensity (x 10$^{4}$ cts/mW/s)',fontsize=label_s)
    plt.tick_params(direction='in',labelsize=label_s,length=8,width=3,labelcolor='black', colors='black', which='both',top=False,right=True)
    plt.xlim(xlim[0],xlim[1])
    plt.subplots_adjust(top=0.98,bottom=0.14,left=0.095,right=0.93,hspace=0.2,wspace=0.2)
    if fill:
        plt.fill_between(Wav_SERS_633, Spe_SERS_633/10000*mg_633,np.zeros(shape=len(Spe_SERS_633)),alpha=0.5, color='C0')
        plt.fill_between(Wav_SERS_785, Spe_SERS_785/10000*mg_785,np.zeros(shape=len(Spe_SERS_785)),alpha=0.5, color='C1')

def plt_water_fall( Wav, Spe, mg=[100,10**-4,10**-3],fall_level=[1,0.8,1.5], range=[600, 700]):
    i_0=np.argmin(abs(Wav[0]-range[0]))
    i_1=np.argmin(abs(Wav[0]-range[1]))
    Max_i=[Wav[0][np.argmax(i[i_0:i_1])+i_0] for i in Spe[0]]
    sort_index=np.argsort(Max_i)
    
    for i,j in enumerate(Spe[0]):
        if i == 0:
            pf.plt_plot(Wav[0], Spe[0][sort_index[i]]*mg[0],figsize=[8,12],color='black',lineshape='--')
        else:
            plt.plot(Wav[0], Spe[0][sort_index[i]]*mg[0]+i*fall_level[0],'--',color='black',linewidth=3)
    plt.ylabel('Scattering intensity (%)')
    plt.xlabel('Wavelength / nm')

    for i,j in enumerate(Spe[1]):
        if i == 0:
            plt.twinx()
            plt.plot(Wav[1], Spe[1][sort_index[i]]*mg[1],linewidth=1,color='C0')
        else:
            plt.plot(Wav[1], Spe[1][sort_index[i]]*mg[1]+i**fall_level[1],linewidth=1,color='C0')

    for i,j in enumerate(Spe[2]):
        if i == 0:
            plt.twinx()
            plt.plot(Wav[2], Spe[2][sort_index[i]]*mg[2],linewidth=1,color='C1')
        else:
            plt.plot(Wav[2], Spe[2][sort_index[i]]*mg[2]+i**fall_level[2],linewidth=1,color='C1')

    plt.tick_params(labelsize=16)
    plt.ylabel('SERS intensity (x10$^{5}$ cts/mW/s)',fontsize=16)
    plt.tight_layout()

def Normalize_mean(data):
    data_norm=[]
    Dim=len(list(np.shape(data)))
    if Dim == 2:
        for i in range(len(data)):
            norm_v0=(data[i]-np.min(data[i]))/(np.max(data[i])-np.min(data[i]))
            data_norm.append(norm_v0)
        data_norm=np.asarray(data_norm)
    elif Dim == 1:
        data_norm=(data-np.min(data))/(np.max(data)-np.min(data))
    else:
        raise Exception ('The data are not one or two-dimensional array. You need to change the function.')

    return data_norm

def histogram(Pos, width=5,color='C0',plot=True):
    bins_num=int((np.max(Pos)-np.min(Pos))/width)
    counts_pos, bins_pos = np.histogram(Pos,bins=bins_num)
    resonance_pos, fwhm_pos, best_fit_pos, best_fit_x_pos=DFS.Gaussian(counts_pos,bins_pos[:-1]+0.5*(bins_pos[1]-bins_pos[0]))
    if plot:
        plt.hist(bins_pos[:-1], bins=bins_pos, weights=counts_pos,color=color, ec='black',linewidth=1,alpha=0.8)
        plt.plot(best_fit_x_pos,best_fit_pos,'k--',linewidth=2,color='black')
    return resonance_pos


def plt_z_shift(Wav,  data, x_cut=[450,1000],xlim=[500,950],ylim=[0,0],name='',legend=False,smooth=False,return_df=False):
    
    data=np.asarray(data)
    # max_v=[np.max(i[200:300]) for i in data]
    # max_i_0=np.argmax(max_v)

    # y_ticks_z_shift= np.arange(-3*5,2*5,5)
    y_ticks_z_shift= np.round(np.arange(-0.5+0.035,0.6,0.5-0.035),2)
    y_ticks=np.arange(-0.5+0.035,0.6,0.5-0.035)/0.035-np.min(np.arange(-0.5+0.035,0.6,0.5-0.035)/0.035)

    i_min=np.argmin(abs(Wav-x_cut[0]))
    i_max=np.argmin(abs(Wav-x_cut[1]))
    data=data[:,i_min:i_max]
    Wav=Wav[i_min:i_max]
    indx=np.argmax(data,axis=0)
    DF_spe=np.max(data,axis=0)
    pf.plt_contourf(Wav, np.arange(0,len(data)+1,1),data*100,color_range=[0.1,1],figsize=[7,5],pcolormesh=True,cmap='inferno', label_s=18)
    # plt.scatter(Wav, indx, color='white')
    plt.yticks(y_ticks, y_ticks_z_shift)
    plt.subplots_adjust(top=0.96,bottom=0.13,left=0.165,right=0.97,hspace=0.2,wspace=0.2)
    plt.ylabel('Foucs (a.u.)')

    pf.plt_plot(Wav,DF_spe*100,linewidth=3,box_lw=3, label_s=21)
    plt.subplots_adjust(top=0.97,bottom=0.14,left=0.125,right=0.95,hspace=0.2,wspace=0.2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering intensity (%)')
    # for i,j in enumerate(data):
    #     if smooth:
    #         spe = DFS.reduceNoise(j)
    #     else:
    #         spe=j
    #     if i == 0:
    #         pf.plt_plot(Wav,spe*100,leg_label=i,linewidth=1.5)
    #     else:
    #         pf.plt.plot(Wav,spe*100,label=i)
    # if legend:  
    #     plt.legend(fontsize=10)
    # plt.xlim(xlim[0]-10,xlim[1]+10)
    # plt.subplots_adjust(top=0.98,bottom=0.115,left=0.12,right=0.965,hspace=0.2,wspace=0.2)
    # if ylim[0] != ylim[1]:
    #     plt.ylim(ylim[0],ylim[1])
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Scattering intensity (%)')
    if name != '':
        plt.savefig(name+'_0.png',dpi=300, transparent=True)
        plt.close()
        plt.savefig(name+'_1.png',dpi=300, transparent=True)
    if return_df:
        return Wav, np.max(data,axis=0)

def gaussian(x, amp, sigma, cen):
    return amp/(sigma*np.sqrt(2*3.14)) * (np.exp(-(x-cen)**2 / (2*(sigma)**2)))

def Gaussian(x,data,R_TH=0.5,check=False, extend=False):

    GM=GaussianModel(prefix='Gaussian_')
    pars=GM.guess(data,x=x)
    
    init = GM.eval(pars, x=x)
    out = GM.fit(data, pars, x=x)
    
    resonance = out.params['Gaussian_center'].value
    stderr_res = out.params['Gaussian_center'].stderr
    fwhm = out.params['Gaussian_fwhm'].value
    stderr_fwhm = out.params['Gaussian_fwhm'].stderr
    sigma = out.params['Gaussian_sigma'].value
    amp = out.params['Gaussian_amplitude'].value
    height= out.params['Gaussian_height'].value
    stderr_ht = out.params['Gaussian_height'].stderr
    correlation_matrix = np.corrcoef(data, out.best_fit)
    correlation = correlation_matrix[0,1]
    R_squared = correlation**2

    x_new = np.linspace(np.min(x), np.max(x),500)
    y_new = gaussian(x_new, amp, sigma, resonance)
    extend_data=[x_new, y_new]
    if check:
        plt.subplots()
        plt.plot(x,data)
        plt.plot(x,out.best_fit)
        print(R_squared)

    if R_squared < R_TH or out.success == False:
        raise Exception('Fitting fails or R-square is below 0.5 !')
    res ={'Intensity': height,'Intensity_error': stderr_ht,'Position':resonance, 'Position_error':stderr_res,
          'fwhm':fwhm,'Fitted curve':out.best_fit, 'Extending curve':extend_data}

    return res

def Extract_ERS(data,pos,R_TH=0.5):
    data_0=AP.Run(data[pos-30:pos+30], 1, Max_Steps=0.5)
    res = Ff.Gaussian_fit(data_0,range(len(data[pos-30:pos+30])),R_TH=R_TH,check=True)
    ERS = (data[pos-30:pos+30] - data_0)[int(res[0])]
    ratio=res[1]/ERS
    print('ERS: '+str(ERS),'ratio: '+str(ratio), 'Raman: '+str(res[1]))


class SERS_int_extraction:

    def  __init__(self):
        self.file_path=''
        self.scan_num=[]
        self.data_name=['']
        self.data_name_DF=['']
        self.data_name_image=['']
        self.data_name_SERS_bgd=['']
        self.peak_pos=[]
        self.power=[]
        self.exposure=[]
        self.IRS = []
        self.file_path_SERS_bgd=''
        self.SERS_bgd_cutting=None
        self.DF_smooth=False
        self.bgd_path=''
        self.DF_bgd_factor=1
        self.file_path_DF=''
        self.rf_path=''
        self.rf_path_bgd=''
        self.Wav=None
        self.x_range=[550,1000]
        self.notch_region=[617,640,772,792]
        self.data_flip=False

    def params_check(self, param):
        if param == []:
            param = [1 for i in range(len(self.data_name))]
        return param

    def SERS_bgd(self):
        SERS_bgd=[]
        cut_i = self.SERS_bgd_cutting
        with h5py.File(self.file_path_SERS_bgd,'r') as File:
            if len(self.data_name_SERS_bgd) > 1:
                for i,j in enumerate(self.data_name_SERS_bgd):
                    locals()['SERS_bgd_'+str(i)] = np.asarray(File[j])
                    if len(np.shape(locals()['SERS_bgd_'+str(i)])) == 2:
                        if cut_i is not None:
                            locals()['SERS_bgd_'+str(i)]= np.mean(locals()['SERS_bgd_'+str(i)][cut_i[0]:cut_i[1]],axis=0)
                        else:
                            locals()['SERS_bgd_'+str(i)]= np.mean(locals()['SERS_bgd_'+str(i)],axis=0)
                    SERS_bgd.append(locals()['SERS_bgd_'+str(i)])
            else:
                SERS_bgd.append(np.mean(np.asarray(File[self.data_name_SERS_bgd[0]]),axis=0))
        return SERS_bgd

    def Sorting_DF(self):
        with h5py.File(self.file_path_DF,'r') as File:
            bgd=np.asarray(File[self.bgd_path])
            ref=np.asarray(File[self.rf_path])
            ref_bgd=np.asarray(File[self.rf_path_bgd])
        if len(np.shape(bgd)) == 2:
            bgd=np.mean(bgd, axis=0)
        if len(np.shape(ref)) == 2:
            ref=np.mean(ref,axis=0)
        if len(np.shape(ref_bgd)) == 2:
            ref_bgd=np.mean(ref_bgd,axis=0)
        ref=ref-ref_bgd
        bgd=bgd*self.DF_bgd_factor
        return bgd, ref
    
    def DF_reshape(self,Spe_DF):
        index_0=np.argmin(abs(self.x_range[0]-self.Wav))
        index_1=np.argmin(abs(self.x_range[1]-self.Wav))
        Wav=self.Wav[index_0:index_1]
        index_2=np.argmin(abs(self.notch_region[0]-Wav))
        index_3=np.argmin(abs(self.notch_region[1]-Wav))
        index_4=np.argmin(abs(self.notch_region[2]-Wav))
        index_5=np.argmin(abs(self.notch_region[3]-Wav))

        Spe_DF_all=[]
        for i,j in enumerate(Spe_DF):
            if self.DF_smooth:
                Spe_DF_0=DFS.reduceNoise(j[index_0:index_1])
            else:
                Spe_DF_0=j[index_0:index_1]
            Spe_DF_1=np.concatenate((Spe_DF_0[:index_2],np.zeros(len(Spe_DF_0[index_2:index_3])), Spe_DF_0[index_3:index_4],np.zeros(len(Spe_DF_0[index_4:index_5])), Spe_DF_0[index_5:]),axis=0)
            Spe_DF_all.append(Spe_DF_1)
        return Wav, Spe_DF_all

    def Run(self, check=False, DF=True):

        self.power= self.params_check(self.power)
        self.IRS= self.params_check(self.IRS)
        self.exposure= self.params_check(self.exposure)
        SERS_bgd=self.SERS_bgd()

        N=len(self.data_name)
        for i in range(N):
            locals()['spe_'+str(i)] = []
        Spe_DF_all=[]
        name_NP=[]
        Img=[]
        with h5py.File(self.file_path,'r') as File:
            for i in File:
                data=File[i]
                if i[0] == 'P' and len(i) > 20 and int(i[20:]) in self.scan_num:
                    n =[len(data[j]) for j in data]
                    n=int(np.average(n))
                    for k in data:
                        if k[0] == 'P' and len(data[k]) >= n:
                            if DF:
                                bgd, ref= self.Sorting_DF()
                                spe_df =np.asarray(data[k][self.data_name_DF])[0]
                                spe_df=(spe_df-bgd)/(ref)
                                if self.data_flip:
                                    spe_df=np.flipud(spe_df)
                                Spe_DF_all.append(spe_df)
                                name_NP.append(k)
                                Img_0= np.asarray(data[k][self.data_name_image])
                                Img.append(Img_0)
                            for h in range(N):
                                spe_h=np.asarray(data[k][self.data_name[h]])
                                if len(np.shape(spe_h)) > 1:
                                    spe_ave=(np.average(spe_h,axis=0) - SERS_bgd[h])/self.power[h]/self.exposure[h]/self.IRS[h]
                                else:
                                    spe_ave=(spe_h - SERS_bgd[h])/self.power[h]/self.exposure[h]/self.IRS[h]
                                if self.data_flip:
                                    spe_ave=np.flipud(spe_ave)
                                locals()['spe_'+str(h)].append(spe_ave)

        Int=[]
        SERS_ave=[]
        fail_ind=[]
        # plt.plot(locals()['spe_'+str(0)][5])
        # plt.plot(locals()['spe_'+str(1)][5])
        # raise Exception('breaking')
        
        for i in range(N):
            Int_0, fail_ind_0=self.peak_fit(locals()['spe_'+str(i)],self.peak_pos[i])
            Int.append(Int_0)

            if i == 0:
                fail_ind.extend(fail_ind_0)
            else:
                fail_ind.extend([j for j in fail_ind_0 if j not in fail_ind])
        fail_ind.sort(reverse=True)
        # print(fail_ind)

        Int_reshape=[]
        Spe_reshape=[] #  SERS spectra
        Spe_DF_all_reshape=[] # DF spectra
        Img_reshape=[]
        name_NP_reshape=[]
        for i in range(N):
            Int_reshape_0=[]
            Spe_reshape_0=[]
            for j in range(len(Int[i])):
                if j not in fail_ind:
                    Int_reshape_0.append(Int[i][j])
                    Spe_reshape_0.append(locals()['spe_'+str(i)][j])
                    if i ==0:
                        if DF:
                            Spe_DF_all_reshape.append(Spe_DF_all[j])
                            Img_reshape.append(Img[j])
                        name_NP_reshape.append(name_NP[j])
            Int_reshape.append(Int_reshape_0)
            Spe_reshape.append(Spe_reshape_0)
            SERS_ave.append(np.mean(Spe_reshape_0, axis=0))

        if self.Wav is not None and DF:
            Wav_df, Spe_DF_all_reshape = self.DF_reshape(Spe_DF_all_reshape)
            Spe_DF_ave=np.mean(Spe_DF_all_reshape, axis=0)
        elif DF:
            Spe_DF_ave=np.mean(Spe_DF_all_reshape, axis=0)
        else:
            Spe_DF_ave=[]

        if self.Wav is not None:
            Wav=self.Wav
            xlim=[550,960]
        else:
            Wav=np.arange(len(SERS_ave[0]))
            Wav_df=np.arange(len(SERS_ave[0]))
            xlim=[Wav[0],Wav[-1]]

        if check:
            if DF:
                DF_SERS_correlation (Wav_df, Spe_DF_ave, Wav, SERS_ave[0], Wav, SERS_ave[1],mg_633=1,mg_785=1,fill=True, xlim=xlim)
                plt.subplots_adjust(top=0.98,bottom=0.13,left=0.110,right=0.910,hspace=0.2,wspace=0.2)
            else:
                for i in range(N):
                    if i ==0:
                        pf.plt_plot(Wav, SERS_ave[i])
                    else:
                        plt.plot(Wav, SERS_ave[i])

            Int_lg=[np.log10(i) for i in Int_reshape]
            Width=[(np.max(Int_lg)-np.min(Int_lg))/20 for i in Int_lg]
            pf.plt_hist_fit_multi(Int_lg, width=Width)
            
            plt.subplots()
            plt.scatter(Int_lg[0],Int_lg[1])

        res_0 = {str(j)+'_intensity':Int_reshape[i] for i,j in enumerate(self.data_name)}
        res_1 = {str(j)+'_average spectra': SERS_ave[i] for i,j in enumerate(self.data_name)}
        res_2 = {str(j)+'_spectra': Spe_reshape[i] for i,j in enumerate(self.data_name)}
        res_3= {'DF spectra': Spe_DF_all_reshape, 'Particle names':name_NP_reshape, 'DF average spectrum':Spe_DF_ave, 
                'DF wavelength': Wav_df, 'SERS wavelength': Wav, 'Images': Img_reshape}
        res={**res_0, **res_1,  **res_2,  **res_3}

        return res

    def peak_fit(self, data, peak_pos):
        i_0=peak_pos-15
        i_1=peak_pos+15
        Int=[]
        fail_ind = []
        for i,j in tqdm(enumerate(data)):
            check=False
            spe=AP.Run(j[i_0:i_1], Degree=1 ,Max_Steps=0.5)
            x, spe_fun=interpolate_fun(range(len(spe)),spe,size=200)
            if i +2 ==len(data)-50:
                check=False
            try:
                res=Gaussian(x,spe_fun,check=check)
                Int.append(res['Intensity'])
            except:
                Int.append(0)
                fail_ind.append(i)
                print('Fitting fails!',i)

        return Int, fail_ind


def data_to_igor(data,name=''):
    data_output=[]
    for i in data:
        img=np.asarray(i)
        data_output.append(np.transpose(img[:,:,0]))
        data_output.append(np.transpose(img[:,:,1]))
        data_output.append(np.transpose(img[:,:,2]))
    data_output=np.asarray(data_output)
    data_output=np.reshape(data_output,(-1,100))
    np.savetxt(name+'.txt',data_output)
    return data_output


