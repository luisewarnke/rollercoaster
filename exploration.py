# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:27:38 2019

@author: Luise Warnke

Functions for statistics and graphics in thesis
"""

import os
#import sys
#import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
import csv
import file_manipulation as files
import re
import scipy

filepath = os.path.abspath(r"C:\Users\louwa\Documents\Python Master\Project Files\data\vr")
filelist = os.listdir(filepath)

output = r"C:\Users\louwa\Documents\Python Master\Project Files\output\quality.csv"
    
def test_quality(filelist):
    with open(output, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Participant", "PPG","Missing Data"])
    ear = []
    finger = []
    for file in filelist:
        if file == "Demographics.csv":
            demographics = files.Demographics(file)
    for file in filelist:
        if file.endswith("PC.csv"):
            raw_data = files.Shimmer(file)
            part = re.search('Session(.+?)_',file).group(1)
            ppg_pos = demographics["PPG"].loc[int(part)]   
            df = raw_data["HR"]
            quality = len(df.loc[(df<0)])/len(df)
            with open(output, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([part, ppg_pos, quality])
            if ppg_pos == 1:
                ear.append(quality)
            else:
                finger.append(quality)
            print("Participant ", part, ":", quality)
            
    print("Finger: ", len(finger), "measured. Quality:", np.mean(finger), np.median(finger), np.std(finger))
    print("Ear: ", len(ear), "measured. Quality:", np.mean(ear),np.median(ear), np.std(ear))
    
    sns.distplot(ear, color="r", label = "Earlobe", norm_hist = True, bins = 10)
    sns.distplot(finger, color="b", label = "Finger", norm_hist = True, bins = 10)
    plt.xlabel("Missing values")
    plt.legend()
    #plt.savefig("quality.png")
    plt.show()
           
def test_segments(filepath):
    data = files.load(os.path.join(filepath,"Segments.csv"))
    print("__FULL__")
    tests(data)
    for i in range(1, int(data["Video"].max())+1):
        vid = data[data.Video==i]
        print("__VID "+str(i)+"__")
        tests(vid)

def tests(data):
    var = ["HR_max", "SCR"]
    seg0 = data.where(data.Segment == 0.0).dropna()
    seg1 = data.where(data.Segment == 1.0).dropna()
    seg2 = data.where(data.Segment == 2.0).dropna()
    seg3 = data.where(data.Segment == 3.0).dropna()
    seg4 = data.where(data.Segment == 4.0).dropna()
    seg134 = data.where((data.Segment == 1.0) | (data.Segment == 3.0) | (data.Segment == 4.0)).dropna()
    seg124 = data.where((data.Segment == 1.0) | (data.Segment == 2.0) | (data.Segment == 4.0)).dropna()
    seg123 = data.where((data.Segment == 1.0) | (data.Segment == 2.0) | (data.Segment == 3.0)).dropna()
    seg234 = data.where((data.Segment == 2.0) | (data.Segment == 3.0) | (data.Segment == 4.0)).dropna()
    print("MEANS:")
    print(data.groupby('Segment').mean())
    print("SD:")
    print(data.groupby('Segment').std())
    print("MIN:")
    print(data.groupby('Segment').min())
    print("MAX:")
    print(data.groupby('Segment').max())
    print("H1:")
    t_test(seg4, seg1, var)    
    print("H2:")
    t_test(seg2, seg134, var)
    print("H3:")
    t_test(seg3, seg124, var)  
#    print("BONUS:")
#    t_test(seg4, seg123, var)
#    t_test(seg1, seg234, var)
 
def t_test(group1, group2,var):
    for variable in var:
        if len(group1) > len(group2):
            list1, list2 = omit(group1, group2, variable)        
        else:
            list2, list1 = omit(group2, group1, variable)
        print(variable, ": df = ", len(list2)-1, scipy.stats.ttest_rel(list1,list2))    

def omit(group1, group2, var):
    list1 = []
    list2 = []
    for i in range(1, int(group1["Participant"].max()+1)):
        for j in range(1,int(group1.Video.max())+1): 
            df1 = group1[(group1["Participant"]==i)&(group1["Video"]==j)]
            df2 = group2[(group2["Participant"]==i)&(group2["Video"]==j)]
            if len(df1) > 0 and len(df2) > 0:
                list1.append(df1[var].max())
                list2.append(df2[var].max())
    return list1,list2

def distance():
    data = files.load("Segments.csv")
    maxval = []
    maxdif = []
    for i in range(1, int(data["Participant"].max()+1)):
        df = data[data["Participant"]==i]
        if len(df) > 0:
            top = df["SCR"].max()
            maxval.append(top)
            base = df[df["Video"]==0]
            maxdif.append(top - base["SCL"].item())
            
    ax = sns.distplot(maxval, bins=13, color = "#feb24c", kde = False, norm_hist = True)
    ax2 = ax.twinx() 
    ax.set_ylabel('Density')
    ax.set_xlabel("Overall maximal HR per participant")
    ax2.set_ylabel('Difference to baseline')
    sns.scatterplot(maxval, maxdif, ax = ax2, color = "#fd8d3c")
    plt.rcParams['figure.figsize']=(10,8)
    #plt.savefig("difmax.png") 
