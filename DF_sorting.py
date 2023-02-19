# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:19:48 2021

@author: sh2065
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from lmfit.models import GaussianModel, LorentzianModel, LinearModel
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


class DF_analysis:

    def __init__(self, file_path=None):
        self.z_scan_name = 'z_scan_0'
        self.img_name = 'CWL.thumb_image_0'
        self.scan_range = [0,1000]
        self.particle_range = [0, np.inf]
        self.spe_range = [490,960]
        self.width_hist=10
        self.bin_threshold=10
        self.label_size=22
        self.file_path = file_path
        self.data_name ='NPoMs'
        self.prominence = 0.0005
        self.width_find_peak = 35
        self.threshold=10
        self.peak_cut=[600, np.inf]
        self.bgd_remove=False
        self.Normal_NPoM=True
        self.Normal_data=False
        self.Wav=[]
        self.ref=[]
        self.bgd=[]

    def Load_data(self):

        if self.file_path is None:
            try:
                File_path=np.loadtxt('File_path.txt',delimiter=',', dtype=np.str)
                print('Loading File...')
            except:
                File_path=input('Please enter your file path: ')
                Data_name=input('Please enter the name of your data: ')
                np.savetxt('File_path.txt',[File_path, Data_name],fmt='%s')
                File_path=np.loadtxt('File_path.txt',delimiter=',', dtype=np.str)
            self.data_name = File_path[1]
        else:
            File_path=[self.file_path]

        with h5py.File(File_path[0],'r') as File:
            container=[]
            name=[]
            name_not_cen=[]
            img_not_cen=[]
            Img_all=[]

            for j, i in enumerate(File):
                if self.Normal_data:
                    if i[0]=='P' and i[-1] !='l' and self.scan_range[0] <= int(i[-1]) <=self.scan_range[1]: 
                        data=File[i]
                        for k, l in enumerate(sorted(data,key=len)):
                            if list(l)[0] == 'P' and len(data[l]) >= 2 and self.particle_range[0] <= int(l[9:]) <= self.particle_range[1]:
                                img_0=data[l][self.img_name]
                                Img_all.append(img_0)
                                c_c=self.check_center(img_0)
                                if c_c == True:
                                    spe=data[l][self.z_scan_name]
                                    container.append(np.max(spe,axis=0))
                                    if self.Wav == []:
                                        Wav=np.array(spe.attrs['wavelengths']).astype(np.float64)
                                        Ref=np.array(spe.attrs['reference']).astype(np.float64)
                                        Bgd=np.array(spe.attrs['background']).astype(np.float64)
                                    else:
                                        Wav=self.Wav
                                        Ref=self.Ref
                                        Bgd=self.Bgd
                                    name.append(str(l) + '_Scan_'+i[-1])
                                else:
                                    name_not_cen.append(str(l)+ '_Scan_'+i[-1])
                                    img_not_cen.append(img_0)

                elif self.img_name =='':
                    if i[0]=='P' and i[-1] !='l' and self.scan_range[0] <= int(i[-1]) <=self.scan_range[1]: 
                        data=File[i]
                        for k, l in enumerate(sorted(data,key=len)):
                            if list(l)[0] == 'P' and len(data[l]) >= 1 and self.particle_range[0] <= int(l[9:]) <= self.particle_range[1]:
                                spe=data[l][self.z_scan_name]
                                container.append(np.max(spe,axis=0))
                                Wav=np.array(spe.attrs['wavelengths']).astype(np.float64)
                                Ref=np.array(spe.attrs['reference']).astype(np.float64)
                                Bgd=np.array(spe.attrs['background']).astype(np.float64)
                                name.append(str(l) + '_Scan_'+i[-1])
                else:
                    for k in File[i]:
                        if k == self.z_scan_name:
                            spe=File[i][k]
                            Wav=np.array(spe.attrs['wavelengths']).astype(np.float64)
                            Ref=np.array(spe.attrs['reference']).astype(np.float64)
                            Bgd=np.array(spe.attrs['background']).astype(np.float64)
                            container.append(np.mean(spe,axis=0))
                            name.append(i)

            print(str(len(name_not_cen)) + ' Particles are not at the center...')
            
            print(np.shape(container))
            DF_spe=[(j-Bgd)/(Ref-Bgd) for j in container]
            data_spectra=np.asarray(DF_spe)
            
            index_1=np.argmin(abs(Wav-self.spe_range[0]))
            index_2=np.argmin(abs(Wav-self.spe_range[1]))
            
            Spe=data_spectra[:,index_1:index_2]
            Wav_new=Wav[index_1:index_2]
    
            res = {'Particle name': name, 'Particle (NC)': name_not_cen, 
                   'File path': File_path, 'Background': Bgd, 'Reference': Ref}
    
        return Wav_new, Spe, res

    def check_center(self, image, check=False,pixel_range=[40,60],rd_range=[0,16]):
    
        img_0=np.asarray(image)
        img_cut=img_0[pixel_range[0]:pixel_range[1],pixel_range[0]:pixel_range[1],:]
        imgray = cv.cvtColor(img_cut, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray,self.threshold, 255, cv.THRESH_TOZERO)
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
            ret_w, thresh_w = cv.threshold(imgray_w, self.threshold, 255, cv.THRESH_BINARY)
            contours_w, hierarchy_w = cv.findContours(thresh_w, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            img_ct=cv.drawContours(img_0, contours_w, -1, (0,255,0), 1)
            plt.imshow(img_ct)
            plt.subplots()
            x, y, radius = np.int0((x,y,radius))
            img_c=cv.circle(img_cut, (x,y), radius, (0, 0, 255), 2)
            plt.imshow(img_c)
            print('Radius is:' +str(radius))
        return result

    def Analyse(self, save_data=True, order=1, normalize= False):
        
        Wav, Spe, res_0 = self.Load_data()
        name = res_0['Particle name']
        Pos_max=[]
        Fail_spe=[]
        Fail_name=[]
        Succ_name=[]
        Spe_succ=[]
        index_0=np.argmin(abs(Wav-530))
    
        print('Start analyzing...')
        print('You have '+str(len(Spe))+' spectra!')

        try:
            os.remove('Scattering spectra of ' + str(self.data_name) +'.h5')
            f_1=h5py.File('Scattering spectra of ' + str(self.data_name) +'.h5','w')
        except:
            f_1=h5py.File('Scattering spectra of ' + str(self.data_name) +'.h5','w')
        
        a=f_1.create_group('NPoMs (normal spectra)')
        b=f_1.create_group('NPoMs (Weird spectra)')
        
        for i in tqdm(range(len(Spe))):
            if np.max(Spe[i]) == np.min(Spe[i]):
                Fail_spe.append(Spe[i])
                Fail_name.append(name[i])
                c0=b.create_dataset(name=name[i],data=Spe[i])
                c0.attrs['wavelengths']=Wav
                continue
        
            spe_sms_0=reduceNoise(Spe[i])
            peaks, _ = find_peaks( spe_sms_0, prominence= self.prominence, width=self.width_find_peak)
            if len(peaks) > 0:
                Pos_max_0=Wav[np.max(peaks)]
                if self.Normal_NPoM:
                    if self.peak_cut[1] > Pos_max_0 > self.peak_cut[0] and len(peaks) < 4 and spe_sms_0[peaks[-1]] >= spe_sms_0[index_0]:
                        Pos_max.append(Pos_max_0)
                        Spe_succ.append(Spe[i])
                        Succ_name.append(name[i])
                        c1=a.create_dataset(name=name[i],data=Spe[i])
                        c1.attrs['wavelengths']=Wav
                    else:
                        Fail_spe.append(Spe[i])
                        Fail_name.append(name[i])
                        c0=b.create_dataset(name=name[i],data=Spe[i])
                        c0.attrs['wavelengths']=Wav
                else:
                    Pos_max.append(Pos_max_0)
                    Spe_succ.append(Spe[i])
                    Succ_name.append(name[i])
                    c1=a.create_dataset(name=name[i],data=Spe[i])
                    c1.attrs['wavelengths']=Wav
            else:
                Fail_spe.append(Spe[i])
                Fail_name.append(name[i])
                c0=b.create_dataset(name=name[i],data=Spe[i])
                c0.attrs['wavelengths']=Wav
        
            if i+1 == int(len(Spe)):
                print('All finished! (100%)')
        print(peaks, _)
        f_1.close()
        Spe_succ=np.asarray(Spe_succ)
        np.savetxt(self.data_name+'_Useful_para.txt',['Total_spe_num: ' + str(len(Spe)), 'Normal_spe_num: ' + str(len(Spe_succ)), 'Weird_spe_num: ' + str(len(Fail_spe))],fmt='%s')
        
        print('Plotting figures...')
        res_2 = self.plot_hist_spe(Wav, Spe_succ, Pos_max, save_data, self.bgd_remove, order, normalize)

        res_1 = {'Wavelengths': Wav,'Fail spectra': Fail_spe, 'Fail name': Fail_name, 'Succ name': Succ_name, 'Succ spectra': Spe_succ}

        res = {**res_0,**res_1, **res_2}

        return Pos_max, res

    def plot_hist_spe(self, Wav, spectra, pos, save_data=True, bgd_remove=False, order=1, normalize= False):
    
        bins=int((np.max(pos)-np.min(pos))/ self.width_hist)
        Pos_3=pos
        counts_pos, bins_pos = np.histogram(Pos_3,bins=bins)
        figsize=10,7
        fig, ax = plt.subplots(figsize=figsize)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
    
        dark_jet = cmap_map(lambda x: x*0.8, matplotlib.cm.jet)
        n, b, patches = plt.hist(bins_pos[:-1], bins=bins_pos, weights=counts_pos,color='grey', ec='black',linewidth=1,alpha=1)
    
        b_centers = 0.5 * (b[:-1] + b[1:])
        b_c_new=[]
        patches_new=[]
        for i in range(len(counts_pos)):
            if counts_pos[i] >= self.bin_threshold:
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
        resonance_pos, fwhm_pos, best_fit_pos, best_fit_x_pos, std=Gaussian(counts_pos,bins_pos[:-1]+0.5*(bins_pos[1]-bins_pos[0]))
        plt.plot(best_fit_x_pos,best_fit_pos,'k--',linewidth=3)
        plt.tick_params(direction='in', length=7, width=3, labelcolor='black', colors='black', which='both',top=False,left=False)
        plt.xlim(self.spe_range[0]-10,self.spe_range[1]+10)
        plt.xlabel('Wavelength (nm)', multialignment='center', rotation=0, fontsize=self.label_size)
        plt.ylabel('Frequency', multialignment='center', rotation=-90, fontsize=self.label_size,labelpad=25)
        plt.tick_params(labelsize=self.label_size,pad=9)
        
        index_0=np.argmin(abs(Wav-500))
        index_1=np.argmin(abs(Wav-620))
        index_2=np.argmin(abs(Wav-900))
        index_3=np.argmin(abs(Wav-470))
        index_4=np.argmin(abs(Wav-530))
        ax2=ax.twinx()
        ax2_y_max=[]
        ax2_y_min=[]
        bins_spe=[]
        for i in range(len(bins_pos[:-1])):
            aver_spe_b=[]
            for j in range(len(Pos_3)):
                if bins_pos[i]<= Pos_3[j] < bins_pos[i+1]:
                    aver_spe_b.append(spectra[j])
    
            if len(aver_spe_b)>=self.bin_threshold:
                aver_spe_b=np.mean(aver_spe_b,axis=0)
                aver_spe_b=reduceNoise(aver_spe_b, cutoff = 1000, fs = 80000, factor = 7)
                if bgd_remove:
                    aver_spe_b=IP.Run(Wav,aver_spe_b,Poly_Order=order,Maximum_Iterations=5)
                    aver_spe_b=aver_spe_b-np.min(aver_spe_b[index_3:index_4])
                if normalize:
                    aver_spe_b= aver_spe_b/(np.max(aver_spe_b[index_0:index_1]))
                ax2_y_max.append(np.max(aver_spe_b[index_0:index_2])) 
                ax2_y_min.append(np.min(aver_spe_b[index_0:index_2]))
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
            ax2.set_ylabel('Scattering intensity', multialignment='center', rotation=90, fontsize=self.label_size)
        else:
            ax2.set_ylabel('Scattering intensity (%)', multialignment='center', rotation=90, fontsize=self.label_size)
        if normalize: 
            ax2.set_ylim(np.min(ax2_y_min)-abs(np.min(ax2_y_min)*0.2),np.max(ax2_y_max)*1.2)
        else:
            ax2.set_ylim(np.min(ax2_y_min)*100-abs(np.min(ax2_y_min)*100*0.2),np.max(ax2_y_max)*1.2*100)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        plt.tick_params(labelsize=self.label_size,pad=9)
        plt.tick_params(direction='in', length=7, width=3, labelcolor='black', colors='black', which='both',top=False,right=False)
        plt.subplots_adjust(top=0.975,bottom=0.115,left=0.155,right=0.875,hspace=0.2,wspace=0.2)
        if save_data:
            np.savetxt(self.data_name+'_pos_hist.txt',['resonance_pos: ' +str(resonance_pos), 'fwhm_pos: '+str(fwhm_pos),'standard dve:'+str(std)],fmt='%s')       
            np.savez(self.data_name+'_All_spe',c_pos=pos,spectra=spectra,Wav=Wav)
            np.savez(self.data_name+'_bin_center_spectra_wid',Wav=Wav,spectra=bins_spe,bins=bins)
            plt.savefig(self.data_name+'_all_hist_bins_wid.jpg',dpi=300)
        plt.show()
        fit_res = [resonance_pos, fwhm_pos, std]
        bin_fit_curve =[best_fit_x_pos, best_fit_pos]
        res={'Fitting reslut': fit_res, 'Bin fitting curve': bin_fit_curve,'Bin wavelengths': Wav, 'Bin spectra': bins_spe }

        return res



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
    std= np.std(out.best_fit)
    
    return resonance, fwhm, out.best_fit, x_new, std

if __name__ == "__main__":
    Analysis=DF_analysis()
    Pos_max, res = Analysis.Analyse()


