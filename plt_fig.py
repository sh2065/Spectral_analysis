# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:21:20 2020

@author: sh2065
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from lmfit.models import GaussianModel, LorentzianModel, ExponentialModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel,  LinearModel, PolynomialModel
from scipy import interpolate, signal
from matplotlib import cm
from matplotlib.widgets import Slider, Button, RadioButtons
from Shu_analysis import DF_sorting as DFS
from mycolorpy import colorlist as mcp

def Linearfit(data,Wav,pass_z=False,extend=False):
    LM=LinearModel(prefix='LM_')
    if pass_z==True:
        Pars=LM.make_params(slope=0)
        for i in range(30):
            Wav=np.insert(Wav,0,0)
            data=np.insert(data,0,0)
    else:
        Pars=LM.guess(data,x=Wav)
        Pars.update(LM.make_params())
    x=Wav
    res=LM.fit(data,Pars,x=Wav)
    best_fit=res.best_fit
    
#    if len(data)<200:
#        x_new=np.linspace(np.min(Wav),np.max(Wav),200)
#        print(x)
#        y=interpolate.interp1d(x,res.best_fit,kind='quadratic')
#        best_fit=y(x_new)
#        x=x_new
#    else:
#        best_fit=res.best_fit

    intercept= res.params['LM_intercept'].value
    slope= res.params['LM_slope'].value
    print(res.fit_report())
    
    if extend:
        n=int (len(Wav) /2)
        interval=(np.max(Wav)-np.min(Wav))/(2*n)
    else:
        n=0
        interval=0
    x_new=np.linspace(np.min(Wav)-interval*n,np.max(Wav)+interval*n,50)
    y_new=slope*x_new + intercept
    best_fit, x = y_new, x_new

    return best_fit, x

def fig_para(name='fig_parameters'):
    
    try:
        xylabel=np.loadtxt(name+'.txt',dtype='str',delimiter=',')
        xlabel=xylabel[0]
        ylabel=xylabel[1]
        xlabel_s=xylabel[2]
        ylabel_s=xylabel[3]
    except:
        print('Collecting the parameters...')
        xlabel=input('Please input the x-label:')
        ylabel=input('Please input the y-label:')
        xlabel_s=input('Please input the size of x-label:')
        ylabel_s=input('Please input the size of y-label:')
        save_label=input('Do you want to save the labels? y/n ')
    
    try:
        if save_label =='y':
            np.savetxt(name+'.txt',[xlabel,ylabel,xlabel_s,ylabel_s],fmt='%s')
    except:
        print('loading the fig_parameters...')
        
    return xlabel,ylabel,xlabel_s,ylabel_s

def plt_normal_t_p(x_axis=[-1,7],left_label=False, bottom_label=False, label_size=16):
    
    plt.xlim(x_axis[0],x_axis[1])
    plt.tick_params(direction='out', length=5, width=2, labelcolor='black', colors='black', which='major', top=False,right=False,bottom=bottom_label,labelsize=label_size)
    plt.tick_params(axis='x',which='both',bottom=bottom_label,top=False,labelbottom=bottom_label,right=False,left=left_label)


def Gaussian_for_hist(data,x):
    n=len(data)   
    x_new=np.linspace(np.min(x),np.max(x),5*n)
    y=interpolate.interp1d(x,data,kind='quadratic')
    data_new=y(x_new)   
    # inserting two 0 on each side of the y
    x_new_1=x_new
    for i in range(100):
        x_new_1=np.insert(x_new_1,0,np.min(x_new_1)-(x_new[2]-x_new[1]))
        x_new_1=np.insert(x_new_1,1+np.argmax(x_new_1),np.max(x_new_1)+(x_new[2]-x_new[1]))
        data_new=np.insert(data_new,0,0)
        data_new=np.insert(data_new,np.argmax(x_new_1),0)
    GM=GaussianModel(prefix='Gaussian_')   
    pars=GM.guess(data_new,x=x_new_1)
    
    init = GM.eval(pars, x=x_new_1)
    out = GM.fit(data_new, pars, x=x_new_1)
    
    resonance = out.params['Gaussian_center'].value
    stderr_res = out.params['Gaussian_center'].stderr
    fwhm = out.params['Gaussian_fwhm'].value
    stderr_fwhm = out.params['Gaussian_fwhm'].stderr
    sigma = out.params['Gaussian_sigma'].value
    

    return resonance, stderr_res, out.best_fit, x_new_1 ,fwhm


def plt_hist_0(data,width=10,fit_color='black',ticks_l=7,ticks_w=3,line_w=1.5,box_lw=3,figsize=[8,6],x_axis=[0,0],y_axis=[0,0],color='C0', direction='in', label=False,new_fig=False,alpha=1,x_label=False,y_label=False,x_label_s='16',y_label_s='16',fit=True):
    
    if new_fig:
        fig,ax=plt.subplots(figsize=figsize)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(box_lw)
    bins = int((np.max(data)-np.min(data))/width)
    counts,bins=np.histogram(data,bins=bins)
    resonance, error_bar, fit_curve, fit_curve_x, fwhm= Gaussian_for_hist(data=counts,x=bins[:-1]+0.5*(bins[1]-bins[0]))
    if x_axis[1]==0 and y_axis[1]==0:
        print('Free axis')
        plt.hist(bins[:-1],bins,weights=counts,ec='black',linewidth=0.5,color=color,alpha=alpha,label=label)
        if fit==True:
            plt.plot(fit_curve_x,fit_curve,'--',color=fit_color,linewidth=line_w)
    elif x_axis[1]==0:
        print('Free x_axis')
        plt.hist(bins[:-1],bins,weights=counts,ec='black',linewidth=0.5,color=color,alpha=alpha,label=label)
        if fit==True:
            plt.plot(fit_curve_x,fit_curve,'--',color=fit_color,linewidth=line_w)
        plt.ylim(y_axis)
    else:
        print('Free y_axis')
        plt.hist(bins[:-1],bins,weights=counts,ec='black',linewidth=0.5,color=color,alpha=alpha,label=label)
        if fit==True:
            plt.plot(fit_curve_x,fit_curve,'--',color=fit_color,linewidth=line_w)
        print(x_axis)
        plt.xlim(x_axis)        
    if y_label==False:
        plt.tick_params(direction=direction, length=ticks_l, width=ticks_w, labelcolor='black', colors='black', which='major',top=False,right=False,labelsize=int(y_label_s),pad=7.5)
        plt.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False,right=False)
    else:
        plt.tick_params(direction=direction, length=ticks_l, width=ticks_w, labelcolor='black', colors='black', which='major',top=False,right=False,labelsize=int(y_label_s),pad=7.5)
        #plt.ylabel(y_label, multialignment='right', rotation=90, fontsize=int(y_label_s),position=(0.5,0.1))
        plt.ylabel(y_label, multialignment='right', rotation=90, fontsize=int(y_label_s))
    
    if x_label==False:
        plt.tick_params(direction=direction, length=ticks_l, width=ticks_w, labelcolor='black', colors='black', which='major',top=False,right=False,bottom=False,labelsize=int(x_label_s),pad=7.5)
        plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False,right=False)
    else:
        plt.tick_params(direction=direction, length=ticks_l, width=ticks_w, labelcolor='black', colors='black', which='major',top=False,right=False,labelsize=int(x_label_s),pad=7.5)
        #plt.xlabel('(Intensity / (cps/mW)$\mathregular{^{1/6}}$', multialignment='center', rotation=0, fontsize=label_size)
        plt.xlabel(x_label,multialignment='center', rotation=0, fontsize=int(x_label_s))

    if label==False:
        print('No label')
    else:
        plt.legend(loc=1, prop={'size': 20})
    print(fwhm)
    print(resonance)
    return resonance, error_bar

def plt_sca(x,data,error_bar=0,figsize=[4,3],l_wid=3,box_lw=2.5,pointsize=6,x_label=True,y_label=True,label_s=12,newfigs=True,line_color='',edgecolor='none',emp_circle=False,fit=False,pass_z=False, extend=False,alpha=1,pad=7.5, 
            ylabel_pos='left',return_data=False, marker='o'):
    
    if x_label==False:
        xlabel='Range'
    elif x_label==True:
        xlabel='Laser Power (mW)'
    else:
        xlabel=x_label
    if newfigs:
        figsize=figsize
        fig, ax =plt.subplots(figsize=figsize)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(box_lw)
    else:
        print('Combing figs')
    if len(np.shape(error_bar)) >0:
        plt.errorbar(x,data, yerr=np.array(error_bar),fmt='.',ecolor='black',elinewidth=l_wid,zorder=1,color='black')
    color=[]
    for i in range(len(data)):
        color.append('C'+str(i))
    #plt.scatter(x,data,c=color, cmap='tab10',zorder=2,linewidth=pointsize)
    if line_color=='':
        plt.scatter(x,data,zorder=2,linewidth=int(pointsize/20),alpha=alpha, marker=marker)
    else:
        facecolor=line_color
        if emp_circle:
            facecolor='none'
        if alpha == 1 :
            edgecolor='black'
        plt.scatter(x,data,color=line_color,zorder=2,s=pointsize, facecolors=facecolor, edgecolors=edgecolor,alpha=alpha, marker=marker)

    if fit==True:
        y,x=Linearfit(data,x,pass_z=pass_z, extend=extend)
        plt.plot(x,y,'--',linewidth=3,color='black')

    if ylabel_pos == 'right':
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y',pad=pad,right=True, left=False,labelleft=False,labelright=True)
    plt.tick_params(direction='in', length=7, width=3, labelcolor='black', colors='black', which='major',top=False,labelsize=label_s,pad=pad)
    plt.ylabel(y_label, rotation=90, fontsize=label_s)
    plt.xlabel(xlabel, multialignment='center', rotation=0, fontsize=label_s)
    plt.subplots_adjust(top=0.975,bottom=0.17,left=0.23,right=0.975,hspace=0.2,wspace=0.2)
    if return_data and fit:
        return x,y

def plt_hist_fit_multi(data,width,fig_size=[8,16],alpha=1,set_label=False,x_axis=[0,0],fit=True):
    
    if set_label==True:
        xlabel,ylabel,xlabel_s,ylabel_s=fig_para()
    else:
        xlabel='(Intensity / (cts/mW/s))$\mathregular{^{1/10}}$'
        ylabel='Frequency'
        xlabel_s='16'
        ylabel_s='16'
        
    plt.subplots(figsize=fig_size)
    Pos=[]
    Pos_error=[]
    for i in range(len(data)):
        if i+1==len(data):
            plt.subplot(len(data),1,i+1)
            Pos_0,Pos_error_0=plt_hist_0(data[i],width=width[i],color='C'+str(i),label=False,alpha=alpha,x_axis=[np.min(data[0])-1,np.max(data[0])+1],x_label=xlabel,y_label=ylabel,fit=fit)
            Pos.append(Pos_0)
            Pos_error.append(Pos_error_0)
        else:
            plt.subplot(len(data),1,i+1)
            Pos_0,Pos_error_0=plt_hist_0(data[i],width=width[i],color='C'+str(i),label=False,alpha=alpha,x_axis=[np.min(data[0])-1,np.max(data[0])+1],x_label=xlabel,y_label=ylabel,fit=fit)
            Pos.append(Pos_0)
            Pos_error.append(Pos_error_0)
    plt_sca(x=range(len(Pos)),data=Pos,error_bar=Pos_error)
    
    
    return Pos,Pos_error

# for plot four dataset

def plt_hist_fit(data, x_lim=[0,0],ticks_l=7,ticks_w=3, width=[0.2,0.2,0.2,0.2],color=[],fig_size=[4.5,7], box_lw=2, direction='in', shape=[4,1],alpha=1,laser_power=[0,0],set_label=False,fit=True, plot=True):
    
    if set_label==True:
        xlabel,ylabel,xlabel_s,ylabel_s=fig_para()
    else:
        xlabel='(Intensity / (cps/mW))$\mathregular{^{1/6}}$'
        ylabel='Frequency'
        xlabel_s='16'
        ylabel_s='16'
    if x_lim[1]==0:
        x_range=[0.8*np.min(data[1]),1.2*np.max(data[1])]
    else:
        x_range=x_lim
    
    Pos=[]
    Error=[]
    N=shape[0]
    M=shape[1]
    if shape[1] > 1:
        plt.subplots(figsize=fig_size)
        n=0
        for i in range(N):
            plt.subplot(N,1,i+1)
            for j in M:
                locals()['Pos_'+str(i+1)+'_'+str(j+1)], locals()['Pos_'+str(i+1)+'_'+str(j+1)+'_error'] = plt_hist_0(data[1],width=width[0],ticks_l=ticks_l,ticks_w=ticks_w, color='C' +str(n),alpha=alpha,x_axis=x_range,x_label_s=xlabel_s,y_label_s=ylabel_s,fit=fit)
                n=n+1
                Pos.append(locals()['Pos_'+str(i+1)+'_'+str(j+1)])
                Error.append(locals()['Pos_'+str(i+1)+'_'+str(j+1)+'_error'])
    elif shape[1] == 1:
        plt.subplots(figsize=fig_size)
        for i in range(N):
            ax=plt.subplot(N,1,i+1)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(box_lw)
            if color == []:
                color_plot = 'C'+str(i)
            else:
                color_plot = color[i]
            if i+1 == N: 
                locals()['Pos_'+str(i+1)],locals()['Pos_'+str(i+1)+'_error']=plt_hist_0(data[i],width=width[i],ticks_l=ticks_l,ticks_w=ticks_w,color=color_plot,alpha=alpha,x_axis=x_range,x_label_s=xlabel_s,y_label_s=ylabel_s,x_label=xlabel,y_label='',direction=direction,fit=fit)
            else:
                locals()['Pos_'+str(i+1)],locals()['Pos_'+str(i+1)+'_error']=plt_hist_0(data[i],width=width[i],ticks_l=ticks_l,ticks_w=ticks_w,color=color_plot,alpha=alpha,x_axis=x_range,x_label_s=xlabel_s,y_label_s=ylabel_s,direction=direction,fit=fit)
            Pos.append(locals()['Pos_'+str(i+1)])
            Error.append(locals()['Pos_'+str(i+1)+'_error'])
        plt.subplots_adjust(top=0.975,bottom=0.105,left=0.195,right=0.945,hspace=0.045,wspace=0.2)
        plt.ylabel(ylabel,position=(0, N/2),fontsize=ylabel_s)
        ax.set_ylabel(ylabel, labelpad=12.5)
    else:
        print('The shape is not included. You could add more!')

    if laser_power[1]==0 and fit==True and plot:
        plt_sca(x=range(len(Pos)),data=Pos,error_bar=Error,x_label=False)
    else:
        if fit==True and plot:
            plt_sca(x=laser_power,data=Pos,error_bar=Error,y_label=xlabel)
    return Pos, Error

def plt_multi_scatter(data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,shape=[4,1],fig_size=[5,7],x_axis=[-1,7],laser_power=0,set_label=False):
    
    if set_label==True:
        xlabel,ylabel,xlabel_s,ylabel_s=fig_para('plt_multi_scatter')
    else:
        xlabel='log(Lifetime / s)'
        ylabel='(Intensity / (cps/mW))$\mathregular{^{1/6}}$'
        xlabel_s='16'
        ylabel_s='16'

    if shape == [4,1]:
        plt.subplots(figsize=fig_size)
        plt.subplot(4,1,1)
        plt.scatter(data_1,data_2,linewidth=1,color='C0')
        plt_normal_t_p(x_axis=x_axis,left_label=False, bottom_label=False,label_size=xlabel_s)
        plt.subplot(4,1,2)
        plt.scatter(data_3,data_4,linewidth=1,color='C1')
        plt.ylabel(ylabel, multialignment='center', rotation=90, fontsize=ylabel_s,position=(0.5,0.1))
        plt_normal_t_p(x_axis=x_axis,left_label=True, bottom_label=False,label_size=xlabel_s)
        plt.subplot(4,1,3)
        plt.scatter(data_5,data_6,linewidth=1,color='C2')
        plt_normal_t_p(x_axis=x_axis,left_label=False, bottom_label=False,label_size=xlabel_s)
        plt.subplot(4,1,4)
        plt.scatter(data_7,data_8,linewidth=1,color='C3')
        plt.xlabel(xlabel, multialignment='center', rotation=0, fontsize=xlabel_s)
        plt_normal_t_p(x_axis=x_axis,left_label=False, bottom_label=True,label_size=xlabel_s)
        plt.subplots_adjust(top=0.98,bottom=0.095,left=0.17,right=0.955,hspace=0.045,wspace=0.2)
    else:
        print('The shape is not included. You could add more!')

    Mean=[np.mean(10**data_1),np.mean(10**data_3),np.mean(10**data_5),np.mean(10**data_7)]
    if laser_power==0:
        plt_sca(x=range(len(Mean)),data=Mean,ylabel=xlabel,xlabel=False)
    else:
        plt_sca(x=laser_power,data=Mean,ylabel='Lifetime / s')

    return Mean

def plt_contourf(x,y,z,set_label=False,x_axis=[0,0],y_axis=[0,0],color_range=[1,1],figsize=[5,8],cmap='inferno',colorbar=False,levels=[0,0], ticks=[0,0],label_s=16,pcolormesh=False):
    
    if set_label==True:
        xlabel,ylabel,xlabel_s,ylabel_s=fig_para('plt_contourf')
    else:
        #xlabel='Ramanshift / cm$\mathregular{^{-1}}$'
        xlabel='Wavelength (nm)'
        ylabel='Time (s)'
        xlabel_s=label_s
        ylabel_s=label_s
    if color_range[0] <= 1 and  color_range[1] <= 1 :
        norm = cm.colors.Normalize(vmin=z.min()*color_range[0], vmax=z.max()*color_range[1])
    else:
        norm = cm.colors.Normalize(vmin=color_range[0], vmax=color_range[1])
    fig, ax = plt.subplots(figsize=figsize)
    if levels[1] == 0:
        levels = np.linspace(z.min()*color_range[0],z.max()*color_range[1],100)
    else:
        levels=levels
    
    if pcolormesh:
        cset1 = ax.pcolormesh(x, y, z, norm=norm, cmap=cmap)
    else:
        cset1 = ax.contourf(x, y, z, 100, levels=levels, norm=norm, cmap=cmap)

    if ticks[1] == 0:
        ticks=np.arange(int(z.min()*color_range[0]),int(z.max()*color_range[1]))
    else:
        ticks=ticks

    if colorbar==True:
        cbar=fig.colorbar(cset1,ticks=ticks,shrink=1,pad=0.05,aspect=40)
        cbar.set_label('SERS (x10$\mathregular{^{4}}$ cts s$\mathregular{^{-1}}$ mW$\mathregular{^{-1}}$)',size=16,rotation=270,labelpad=20)
        #cbar.set_label('Scattering intensity (%)',size=11,rotation=270,labelpad=10)
        cbar.ax.tick_params(labelsize=16)
        print(z.min(),z.max())
    #plt.colorbar(cset1)
    if x_axis[1]!=0:
        ax.set_xlim(x_axis[0], x_axis[1])
    if y_axis[1]!=0:
        ax.set_ylim(y_axis[0], y_axis[1])

    #ax.set_xticks(np.arange(200, 3000, step=500))
    ax.tick_params(direction='in', length=4, width=2, labelsize=xlabel_s,labelcolor='black', colors='white', which='both',top=False,right=False)
    #plt.title('Timer series SERS')
    plt.xlabel(xlabel, multialignment='center', rotation=0, fontsize=xlabel_s)
    plt.ylabel(ylabel, multialignment='center', rotation=90, fontsize=ylabel_s)
    plt.subplots_adjust(top=0.98,bottom=0.095,left=0.17,right=0.965,hspace=0.2,wspace=0.2)

def plt_plot(x,y,figsize=[8,6],ticks_l=8, ticks_w=3, linewidth=3,label_s=16,color=None,z_order=0,alpha=1,box_lw=3,new_fig=True,y_axis='',lineshape='-',leg_label='',pad=7.5):
    xlabel='Wavelength (nm)'
    ylabel='Intensity'
    xlabel_s=label_s
    ylabel_s=label_s
    figsize=figsize
    if new_fig:
        fig,ax=plt.subplots(figsize=figsize)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(box_lw)
        ax.plot(x,y,linewidth=linewidth,color=color,alpha=alpha,ls=lineshape,label=leg_label,zorder=z_order)
        ax.tick_params(direction='in',length=ticks_l,width=ticks_w,labelcolor='black', colors='black', which='both',top=False,right=False,pad=pad)
        if y_axis=='Right':
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    else:
        plt.plot(figsize=figsize)
        plt.plot(x,y,linewidth=linewidth,color=color,alpha=alpha,ls=lineshape,label=leg_label,zorder=z_order)
        plt.tick_params(direction='in',length=ticks_l,width=ticks_w, labelcolor='black', colors='black', which='both',top=False,right=False,pad=pad)

    plt.xlabel(xlabel, multialignment='center', rotation=0, fontsize=xlabel_s)
    plt.ylabel(ylabel, multialignment='center', rotation=90, fontsize=ylabel_s)
    plt.subplots_adjust(top=0.98,bottom=0.115,left=0.12,right=0.965,hspace=0.2,wspace=0.2)
    plt.tick_params(labelsize=xlabel_s)


def plt_active_figs(Wav,spectra,figsize=(8,6)):

    if np.max(Wav) >= 950:
        i0=np.argmin(abs(Wav-450))
        i1=np.argmin(abs(Wav-950))
        Wav=Wav[i0:i1]
        spectra=spectra[:,i0:i1]

    fig, ax =plt.subplots(figsize=figsize)
    plt.subplots_adjust(top=0.96,bottom=0.255,left=0.145,right=0.93,hspace=0.2,wspace=0.2)
    l, = plt.plot(Wav, spectra[0],lw=2)
    plt.xlabel('Wavlength / nm',fontsize=18)
    plt.ylabel('Intensity',fontsize=18)
    plt.tick_params(labelsize=18)

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.14, 0.07, 0.8, 0.05], facecolor=axcolor)

    sfreq = Slider(axfreq, 'NO.', 0, len(spectra)-1, valinit=0, valstep=1)
    sfreq.label.set_size(18)

    def update(val):
        freq = sfreq.val
        l.set_ydata(spectra[freq])
        fig.canvas.draw_idle()

    sfreq.on_changed(update)

    return sfreq

def plt_active_figs_2(Wav,spectra,figsize=(10,8)):

    plt.subplots(figsize=figsize)
    freqs = np.arange(0, len(spectra), 1)
    l, = plt.plot(Wav, spectra[0],lw=2)
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel('Wavlength / nm',fontsize=18)
    plt.ylabel('Intensity',fontsize=18)
    plt.tick_params(labelsize=18)

    class Index:
        ind = 0
        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = spectra[i]
            l.set_ydata(ydata)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = spectra[i]
            l.set_ydata(ydata)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    return bnext, bprev

def plt_z_shift(data_path='',File_path='',x_cut=[450,1000],xlim=[500,950],ylim=[0,0],name='',legend=False,smooth=False,return_df=False):

    with h5py.File(File_path, 'r') as File:
        data=File[data_path]
        Wav=data.attrs['wavelengths']
        ref=data.attrs['reference']
        bgd=data.attrs['background']
        data=np.asarray(data)
    data=data[1:]
    data=(data-bgd)/(ref-bgd)
    i_min=np.argmin(abs(Wav-x_cut[0]))
    i_max=np.argmin(abs(Wav-x_cut[1]))
    data=data[:,i_min:i_max]
    Wav=Wav[i_min:i_max]
    plt_contourf(Wav, np.arange(0,len(data)+1,1),data*100,color_range=[0.4,1],figsize=[7,5],pcolormesh=True,cmap='inferno')
    plt.subplots_adjust(top=0.96,bottom=0.13,left=0.12,right=0.97,hspace=0.2,wspace=0.2)
    plt.ylabel('z shift')
    plt.yticks(np.arange(0,len(data),5))

    for i,j in enumerate(data):
        if smooth:
            spe = DFS.reduceNoise(j)
        else:
            spe=j
        if i == 0:
            plt_plot(Wav,spe*100,leg_label=i,linewidth=1.5)
        else:
            plt.plot(Wav,spe*100,label=i)
    if legend:  
        plt.legend(fontsize=10)
    plt.xlim(xlim[0]-10,xlim[1]+10)
    plt.subplots_adjust(top=0.98,bottom=0.115,left=0.12,right=0.965,hspace=0.2,wspace=0.2)
    if ylim[0] != ylim[1]:
        plt.ylim(ylim[0],ylim[1])
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Scattering intensity (%)')
    if name != '':
        plt.savefig(name+'_0.jpg',dpi=300)
        plt.close()
        plt.savefig(name+'_1.jpg',dpi=300)
    if return_df:
        return Wav, np.max(data,axis=0)

def plt_z_stack(Wav,data,name='',color_range=[1,1],xlim=[0,0], normalize=False):
    plt_contourf(Wav, np.arange(0,len(data),1),data,color_range=color_range,figsize=[6,8],pcolormesh=True,cmap='inferno')
    plt.ylabel('z_stack')
    if xlim[0]!=0:
        plt.xlim(xlim[0],xlim[1])
    if name != '':
        plt.savefig(name+'_0.jpg',dpi=300)
        plt.close()
    if normalize:
        data_norm=[i/np.max(i[200:]) for i in data]
        data_norm=np.asarray(data_norm)
        plt_contourf(Wav, np.arange(0,len(data_norm),1),data_norm,color_range=color_range,figsize=[6,8],pcolormesh=True,cmap='inferno')
        if name != '':
            plt.savefig(name+'_1.jpg',dpi=300)
            plt.close()

def color_segmentation(color_name='Blues', steps=12):
    colors=mcp.gen_color(cmap=color_name,n=steps)
    return colors

def save_png(name,dpi=300):
    plt.savefig(name+'.png',dpi=dpi, transparent=True)
