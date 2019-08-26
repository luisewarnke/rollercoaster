# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:20:03 2019

@author: louwa
"""

import heartpy as hp

def check_quality(data):
    return (len(data.loc[(data<40)])+len(data.loc[(data>220)]))/len(data) > 0.10

def replace(data, sampleRate):
    working_data, measures = hp.process(data, sampleRate, bpmmax=220)
    peaks = working_data['peaklist']
    
    prev_loc = 0
    hr_list = []
    ibi_list = []
    
    for loc in peaks:
        frame = loc - prev_loc
        
        hr = 60/(frame/sampleRate)
        hr_list.extend([hr]*frame)
        
        ibi = (frame/sampleRate)*1000
        ibi_list.extend([-1.0]*(frame-1)+[ibi])
        
        prev_loc = loc
        
    hr_list.extend([hr]*(len(data) - prev_loc))
    ibi_list.extend([-1.0]*(len(data) - prev_loc))
    
    return hr_list, ibi_list
