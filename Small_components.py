    # -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:39:47 2020

@author: sh2065
"""
import h5py
import numpy as np


def input_path(Data_path=True,Name='Path'):
    
    try:
        Path=np.loadtxt(str(Name)+'.txt',delimiter=',', dtype=np.str)
        File_path=Path[0]
        Data_path=Path[1]
    except:
        File_path=input('Please copy your file path here: ')
        if Data_path == True:
            Data_path=input('Please copy your data path here: ')
        else: 
            Data_path=None
        np.savetxt(str(Name)+'.txt',[File_path,Data_path],fmt='%s')
    File=h5py.File(File_path, 'r')
    
    return File, Data_path

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

def Subs(string):
    subs_str='$\mathregular{^{' + string +'}}$'
    return subs_str
    
def Cut_spe(data,Range=[200,2300]):
    index_0=np.argmin(abs(data-Range[0]))
    index_1=np.argmin(abs(data-Range[1]))
    if len(np.shape(data)) == 1:
        data_cut=data[index_0:index_1]
    else:
        data_cut=data[:,index_0:index_1]
    return data_cut, index_0, index_1

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


def remove_PP(data,factor=8):
    data_new=[]
    for i in range(len(data)):

        if np.max(data[i]) > np.mean(data[i]) + factor*np.std(data[i]):
            max_v=data[i][np.argmax(data[i])]
            min_v=data[i][np.argmin(data[i])]
            if 0.5*(max_v-min_v)>(data[i][np.argmax(data[i])-1]-min_v) or 0.5*(max_v-min_v)>(data[i][np.argmax(data[i])+1]-min_v):
                data_l=list(data[i])
                data_l.pop(np.argmax(data[i]))
                data_l.insert(np.argmax(data[i]),np.mean(data[i]))
                try:
                    data_l.pop(np.argmax(data[i])+1)
                    data_l.insert(np.argmax(data[i])+1,np.mean(data[i]))
                except:
                    pass
                data_l.pop(np.argmax(data[i])-1)
                data_l.insert(np.argmax(data[i])-1,np.mean(data[i]))
                data_new.append(np.asarray(data_l))
            else:
                data_new.append(data[i])
        else:
            data_new.append(data[i])
    
    data_new=np.asarray(data_new)
    return data_new