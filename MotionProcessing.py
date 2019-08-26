# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:00:50 2019

@author: louwa
"""

import numpy as np
import pandas as pd
import file_manipulation as files
import scipy.signal as scisig
import os
import seaborn as sns
import matplotlib.pyplot as plt

#filepath = r"C:\Users\louwa\Documents\Python Master\Project Files\data\clean\Coaster5\Motion.csv"

def rotate(AccX, AccY, AccZ, pitch, yaw, roll):
    top, mid, bot = e2r(pitch, yaw, roll)
    
    vec = np.array([AccX, AccY, AccZ])
        
    x = sum([p*q for p,q in zip(top, vec)])
    y = sum([p*q for p,q in zip(mid, vec)])
    z = sum([p*q for p,q in zip(bot, vec)])
    return x,y,z

def prepare_motion(data, sampleRate):
    for col in data.columns:
        data[col] = files.butter_lowpass_filter(data[col], 1, sampleRate)
    data["GyroX"] = data["GyroX"] + 0.56
    data["GyroY"] = data["GyroY"] - 2
    data["GyroZ"] = data["GyroZ"] + 0.79
    data["accAngleX"] = (np.arctan(data["WideY"] / np.sqrt(data["WideX"]**2 + data["WideZ"]**2)) * 180 / np.pi) - 0.58 #0.58 error
    data["accAngleY"] = (np.arctan(-1 * data["WideX"] / np.sqrt(data["WideY"]**2 + data["WideZ"]**2)) * 180 / np.pi) + 1.58 #error
    gyroAngleX = [0]
    gyroAngleY = [0]
    yaw = [0]
    for i in range(1, len(data)):
        gyroAngleX.append(gyroAngleX[i-1]+data["GyroX"].iloc[i].item()*(1/sampleRate))
        gyroAngleY.append(gyroAngleY[i-1]+data["GyroY"].iloc[i].item()*(1/sampleRate))
        yaw.append(yaw[i-1]+data["GyroZ"].iloc[i].item()*(1/sampleRate))
    data["gyroAngleX"] = np.asarray(gyroAngleX)
    data["gyroAngleY"] = np.asarray(gyroAngleY)
    data["yaw"] = np.asarray(yaw)
    data["roll"] = 0.96 * data["gyroAngleX"] + 0.04 * data["accAngleX"]
    data["pitch"] = 0.96 * data["gyroAngleY"] + 0.04 * data["accAngleY"]
    AccX = []
    AccY = []
    AccZ = []
    for i in range(0, len(data)):
        x,y,z = rotate(data["WideX"].iloc[i].item(), 
                       data["WideY"].iloc[i].item(),
                       data["WideZ"].iloc[i].item(), 
                       data["pitch"].iloc[i].item(),
                       data["yaw"].iloc[i].item(),
                       data["roll"].iloc[i].item())
        AccX.append(x)
        AccY.append(y)
        AccZ.append(z)
        #print((i/len(data))*100,"%")
        
    data = data.assign(AccX=AccX,AccY=AccY,AccZ=AccZ)
    data = data[["AccX","AccY","AccZ"]]
    for col in data.columns:
        data[col] = files.butter_lowpass_filter(data[col], 1.0, sampleRate)
    
    
    return data
       
def distance(df): 
    covX = [0]
    covY = [0]
    covZ = [0]
    for i in range(1, len(df)):
        covX.append(covX[i-1]+df["DisX"].iloc[i].item())
        covY.append(covY[i-1]+df["DisY"].iloc[i].item())
        covZ.append(covZ[i-1]+df["DisZ"].iloc[i].item())
    
    df = df.assign(X=covX,Y=covY,Z=covZ)
    
    sns.lineplot(df.index,df["X"].rolling(64).mean())
    sns.lineplot(df.index,df["Y"].rolling(64).mean())
    sns.lineplot(df.index,df["Z"].rolling(64).mean())
    sns.lineplot(df.index,df.Acc.rolling(64).mean())

def motion(data, sampleRate):
    for column in data[["AccX", "AccY", "AccZ"]]:  
        data["Vel"+column[-1]] = data[column]*(1/sampleRate)
        data["Dis"+column[-1]] = data["Vel"+column[-1]]*(1/sampleRate)    
    data["Acc"] = ((data["AccX"]**2+data["AccY"]**2+data["AccZ"]**2)**0.5)
    return data

def e2r(pitch, yaw, roll):
    yawMatrix = np.matrix([[np.cos(yaw), -np.sin(yaw), 0], 
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])

    pitchMatrix = np.matrix([[np.cos(pitch), 0, np.sin(pitch)],
                              [0, 1, 0],
                              [-np.sin(pitch), 0, np.cos(pitch)]])

    rollMatrix = np.matrix([[1, 0, 0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])

    R = yawMatrix * pitchMatrix * rollMatrix
    
    return np.asarray(R[0].tolist()[0]), np.asarray(R[1].tolist()[0]),np.asarray(R[2].tolist()[0])

def filterSignalFIR(data, sampleRate, cutoff=0.4, numtaps=256):
    f = cutoff/(sampleRate/2.0)
    FIR_coeff = scisig.firwin(numtaps,f)
    
    return scisig.lfilter(FIR_coeff,1,data)

def find_start(data, sampleRate, window = 5, shift = 1):
    for col in data.columns:
        data[col] = files.butter_lowpass_filter(data[col], 1, sampleRate)
    try:
        data["Acc"] = ((data["WideX"]**2+data["WideY"]**2+data["WideZ"]**2)**0.5)-9.81
        lim = 1.0
    except:
        data["Acc"] = ((data["AccelX"]**2+data["AccelY"]**2+data["AccelZ"]**2)**0.5)-9.81
        lim = 2.5
#    data["Acc"] = data.Acc.rolling(4*sampleRate).mean()
    found = False
    i=0
    start_time = data.index[0]
    
    while found == False:
        if i*(shift*sampleRate)+(window*sampleRate) < len(data):
            dat = data.iloc[i*(shift*sampleRate):i*(shift*sampleRate)+(window*sampleRate)]
            if data.Acc.iloc[i*(shift*sampleRate)] > lim and abs(dat.Acc.mean()) < 0.5:
                found=True
                return (start_time + pd.to_timedelta((i*shift+int(0.5*window)), unit = 's'))
            else: 
                i+=1
        else:
            return None
            