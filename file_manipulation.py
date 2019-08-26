# -*- coding: utf-8 -*-
"""

Created on Thu Feb  7 10:39:53 2019

@author: luise.warnke

Adaptation of eda_explorer's load_files.py
"""

import pandas as pd
import numpy as np
import scipy.signal as scisig
import datetime
import os

def Shimmer(filepath): 
    # Get the raw data
    data = pd.io.parsers.read_csv(filepath, delim_whitespace = True, skiprows = 3, index_col = [0])

    # Reset the index to be a time and reset the column headers
    if len(data.columns) == 13:
        data.columns = ['AccelX','AccelY','AccelZ','Battery','GSR_Range','EDA','GSR_SR','PPG','IBI','HR','Pres','Temp','Event']
    elif len(data.columns) == 11:
        data.columns = ['AccelX','AccelY','AccelZ','Battery','GSR_Range','EDA','GSR_SR','PPG','IBI','HR','Event']
        
    # Get Start Time
    data.index = pd.to_datetime(data.index, unit = "ms")
    
    # Get sample rate
    n_in_second = len(data.loc[data.index[0]:(data.index[0] + datetime.timedelta(seconds=1))])
    sampleRate = 4 * round(n_in_second/4)
    
#    startTime = data.index[0]    
#    data = interpolateDataTo8Hz(data,sampleRate,startTime)
    
    # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
    data['filtered_eda'] =  butter_lowpass_filter(data['EDA'], 1.0, 256.0, 6)
    
    # Remove Battery, GSR_Range, GSR_SR and Pressure Column; sorted to match eda_explorer
    data = data[['AccelZ','AccelY','AccelX', 'EDA', 'filtered_eda', 'PPG', 'IBI', 'HR', 'Event']]

    return data, sampleRate

def loadData_E4(filepath):
    # Load EDA data
    eda_data = _loadSingleFile_E4(os.path.join(filepath,'EDA.csv'),["EDA"],4)
    # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
    eda_data['filtered_eda'] =  butter_lowpass_filter(eda_data['EDA'], 1.0, 8, 6)

    # Load ACC data
    acc_data = _loadSingleFile_E4(os.path.join(filepath,'ACC.csv'),["AccelX","AccelY","AccelZ"],32)
    # Scale the accelometer to +-2g
    acc_data[["AccelX","AccelY","AccelZ"]] = acc_data[["AccelX","AccelY","AccelZ"]]*(9.81/64.0)

    # Load Temperature data
    hr_data = _loadSingleFile_E4(os.path.join(filepath,'BVP.csv'),["PPG"],64)
    
    data = eda_data.join(acc_data, how='outer')
    data = data.join(hr_data, how='outer')

    # E4 sometimes records different length files - adjust as necessary
    min_length = min(len(acc_data), len(eda_data), len(hr_data))
    
    data = data[['AccelZ','AccelY','AccelX', 'EDA', 'filtered_eda', 'PPG']]

    return data[:min_length], 8

def _loadSingleFile_E4(filepath, list_of_columns, expected_sample_rate):
    # Load data
    data = pd.read_csv(filepath)
    
    # Get the startTime and sample rate
    startTime = pd.to_datetime(float(data.columns.values[0]), unit="s")
    sampleRate = float(data.iloc[0][0])
    data = data[data.index!=0]
    data.index = data.index-1
    
    # Reset the data frame assuming expected_sample_rate
    data.columns = list_of_columns
    if sampleRate != expected_sample_rate:
        print('ERROR, NOT SAMPLED AT {0}HZ. PROBLEMS WILL OCCUR\n'.format(expected_sample_rate))

    # Make sure data has a sample rate of 8Hz
    data = interpolateDataTo8Hz(data,sampleRate,startTime)

    return data

def Acceleration (filepath):
    # Get the raw data
    data = pd.io.parsers.read_csv(filepath, delim_whitespace = True, skiprows = 3, index_col = [0])
    data.columns= ["AccelX", "AccelY", "AccelZ", "WideX", "WideY", "WideZ", "GyroX", "GyroY", "GyroZ", "MagX", "MagY", "MagZ"]
       
    # Get Start Time
    data.index = pd.to_datetime(data.index, unit = "ms") + pd.to_timedelta((1561200986), unit = "s")
    
    # Get sample rate
    #n_in_second = len(data.loc[data.index[0]:(data.index[0] + datetime.timedelta(seconds=1))])
    sampleRate = 16
    
    #startTime = data.index[0]    
    #data = interpolateDataTo8Hz(data,sampleRate,startTime)
  
    # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
    #data['filtered_eda'] =  butter_lowpass_filter(data['EDA'], 1.0, 8.0, 6)
    
    # Remove Battery, GSR_Range, GSR_SR and Pressure Column; sorted to match eda_explorer
    data = data[["WideX", "WideY", "WideZ", "GyroX", "GyroY", "GyroZ"]]

    return data, sampleRate
    
def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y

def load(filepath, index_col = [0], header = [0], parse_dates = True, tag = False):
    data = pd.io.parsers.read_csv(filepath, index_col = index_col, header = header, parse_dates = parse_dates)
    if tag:
        data.columns = ['Tag']
        data.index = pd.to_timedelta(data.index, unit = "s")
    return data

def save(lists, header, index, name):
    array = np.asarray(lists)
    df = pd.DataFrame(array,columns = header)
    df.set_index = df[index]
    df.to_csv(name+".csv")

def save_dict(data, name):
    np.save(name+".npy", data)
    
def load_dict(file):
    data = np.load(file).flat[0]
    return data

def interpolateDataTo8Hz(data,sample_rate,startTime):
    if sample_rate<8:
        # Upsample by linear interpolation
        if sample_rate==2:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='500L')
        elif sample_rate==4:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='250L')
        data = data.resample("125L").mean()
    else:
        if sample_rate>8:
            # Downsample
            idx_range = list(range(0,len(data))) # TODO: double check this one
            data = data.iloc[idx_range[0::int(int(sample_rate)/8)]]
        # Set the index to be 8Hz
        data.index = pd.date_range(start=startTime, periods=len(data), freq='125L')

    # Interpolate all empty values
    data = interpolateEmptyValues(data)
    return data

def interpolateEmptyValues(data):
    cols = data.columns.values
    for c in cols:
        data[c] = data[c].interpolate()

    return data