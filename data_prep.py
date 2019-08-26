# -*- coding: utf-8 -*-
"""
rename to data prep

Created on Thu Feb  7 15:05:53 2019

@author: luise.warnke
"""

import file_manipulation as files
import re
import datetime as dt
import pandas as pd
import EDA_Artifact_Detection as EDA_art
import EDA_Peak_Detection as EDA_peak
import os
import Heart_Rate as HR
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import MotionProcessing as MP
import csv

#filepath = r"C:\Users\louwa\Documents\Python Master\Project Files\data\clean\Coaster4"

snip_len = 3 #sec
start_base = 1 #min

VIDEOS = {1:[1,2,3],
          2:[1,3,2],
          3:[2,1,3],
          4:[2,3,1],
          5:[3,1,2],
          6:[3,2,1]}

COASTER = [1561203024.82, 1561206848.68, 1561210130.25, 1561215096.45, 1561217464.38, 1561218440.75, 1561218760.09]
DUR = [70, 70, 275, 75, 100, 80, 80]

lim_HR = 220

TAGS = {"Start" : 0, "Lift hill" : 1, "First drop" : 2, "Drop" : 3, "Brake" : 4,
        "Hop" : 5, "Tunnel" : 6, "Helix" : 7, "Loop" : 8, "Immelmann" : 9,
        "Corkscrew" : 10, "Roll" : 11, "End brake" : 12, "End" : 13}

def clean_raw(filepath, Shimmer = True, folder = None):  
    tags = {}
    start_times = []
        
    if Shimmer:
        filelist = os.listdir(os.path.abspath(filepath+'\static'))
        for file in filelist:        
            if file.startswith("Tagging"):           
                tagging = files.load(os.path.abspath(filepath+'\static\\'+ file), index_col = [1], tag = True)
                video = re.search('Tagging (.+?)',file).group(1)
                tags[video] = tagging
            elif file.startswith("Demographics"):
                demographics = files.load(os.path.abspath(filepath+'\static\\'+ file),parse_dates = False)
    
    if Shimmer:
        filepath = os.path.abspath(filepath+'\\raw')
        process(filepath, Shimmer, folder, tags, demographics = demographics)
    else:
        filepath = os.path.join(filepath,folder)
        filelist = os.listdir(filepath)
        for file in filelist:
            if file.startswith("Tagging"):
                tags = files.load(os.path.join(filepath,file), index_col = [1], tag = True)
        
        print(folder, filelist)
        for file in filelist:
            if file.startswith("Motion"):
                motion = files.load(os.path.join(filepath,file))
                start_Shimmer = MP.find_start(motion, 16)
                start_times.append(start_Shimmer)
                motion = apply_tags(motion, "park", tags, start_Shimmer)
                motion.to_csv(os.path.join(filepath,file))
                
        if folder == "Baseline":
            start_Shimmer = None
            
        process(filepath, Shimmer, folder, tags, start_Shimmer = start_Shimmer)

def process(filepath, Shimmer, folder, tags, demographics = None, start_Shimmer = None): 
    filelist = os.listdir(filepath)       
    for file in filelist:
        if (file.endswith(".csv") and Shimmer) or file.startswith("Part"):
            if Shimmer:
                part = re.search('Session(.+?)_',file).group(1)
                print("Processing Participant " + str(part))
                raw_data, sampleRate = files.Shimmer(os.path.join(filepath,file))
            else:
                part = re.search('Id(.+?)\.csv',file).group(1)
                raw_data = files.load(os.path.join(filepath,file))
                sampleRate=8
                if folder != "Baseline":
                    start = MP.find_start(raw_data[["AccelX","AccelY","AccelZ"]], sampleRate)
                    if start == None:
                        start = start_Shimmer
                
            '''
            Heart Rate
            '''
            # if more than 10% of the HR are above 220 or below 40 use heartpy for HR+
            print(file, folder)
            hr, ibi = HR.replace(raw_data["PPG"].values, sampleRate)
            raw_data["HR"] = np.asarray(hr)
            raw_data["IBI"] = np.asarray(ibi)
            raw_data.loc[(raw_data['HR']>220) | (raw_data['HR']<40)] = np.nan                
            
            """
            EDA
            """
            labels, raw_data = EDA_art.classify(raw_data, ["Multiclass"], sampleRate)
            if Shimmer:
                EDA_peak.calcPeakFeatures(raw_data,"data\\vr\\peaks\\Peaks_Part"+str(part)+".csv", 1, 0, 2, 2, sampleRate)
            else:
                EDA_peak.calcPeakFeatures(raw_data,"data\\park\\peaks\\"+str(folder)+"\\Peaks_Part"+str(part)+".csv", 1, 0, 2, 2, sampleRate)
            
            if Shimmer:
                condition = demographics["Condition"].loc[int(part)]
                data = apply_tags(raw_data, condition, tags)
            elif folder != "Baseline":
                data = apply_tags(raw_data, "park", tags, start)  
            else:
                raw_data["Tag"] = ""
                data= raw_data
            
            if Shimmer:
                data.to_csv("data\\vr\\clean\\"+'_'.join(["Participant"+str(part),"Condition"+str(int(condition))])+".csv")
            else:
                data.to_csv(os.path.join(filepath,file))
            
            
def segment(filepath, sampleRate, segments, study, Shimmer = True, folder=None):
    tenSec = sampleRate*10
    if Shimmer:
        filelist = os.listdir(os.path.abspath(filepath+'\\clean'))
    else:
        filepath = os.path.join(filepath, folder)
        filelist = os.listdir(filepath)
    headers = ["Participant","Video","Condition","Segment","HR_avg","HR_max","SCR","SCL"]

    var_list = []
    for file in filelist:
        if file.startswith("Part"):  
            try:
                if Shimmer:
                    data = files.load(os.path.abspath(filepath+'\\clean\\'+file))
                    condition = int(re.search('Condition(.+?).csv',file).group(1))
                    part = int(re.search('Participant(.+?)_',file).group(1))
                    print(part, data.columns)
                    peak_data = files.load(os.path.abspath(filepath+'\\peaks\\Peaks_Part'+str(part)+".csv"))
                else:
                    data = files.load(os.path.join(filepath,file))
                    condition = 0                    
                    part = int(re.search('Id(.+?)\.csv',file).group(1))
                    part_name = re.search('Part(.+?)_',file).group(1)
                    peak_data = files.load(os.path.abspath(r"C:\\Users\\louwa\\Documents\\Python Master\\Project Files\\data\\peaks\\"+folder+"\\Peaks_Part"+str(part)+".csv"))

                seg = 0
                for i in segments:   
                    for j in range(0, segments[seg][2]):
                        if segments[seg][0] == "base":
                            video = 0
                            start = start_base*60*sampleRate
                        else:
                            if Shimmer:
                                video = VIDEOS[condition][j]
                            else:
                                video = folder[-1]
                            start = data[data["Tag"]==segments[seg][0]].iloc[j].name
                            try:
                                start = data.index.get_loc(start).start
                            except AttributeError:
                                start = data.index.get_loc(start)
                        end = data[data["Tag"]==segments[seg][1]].iloc[j].name
                        try:
                            end = data.index.get_loc(end).start
                        except AttributeError:
                            end = data.index.get_loc(end)
                        if segments[seg][0] != "base" or Shimmer:
                            frame = data.iloc[start:end]
                            starttime = data.iloc[start].name
                            endtime = data.iloc[end].name 
                            mask = (peak_data.index >= starttime) & (peak_data.index <= endtime)
                            scr_peak = peak_data.loc[mask]
                            scr = (len(scr_peak) / float(len(frame)/sampleRate)) * 60
                        else:
                            frame = files.load(os.path.abspath(r'C:\\Users\\louwa\\Documents\\Python Master\\Project Files\\data\\clean\\Baseline\\Part'+part_name+'_Id'+str(part)+".csv"))
                            base_peak = files.load(os.path.abspath(r'C:\\Users\\louwa\\Documents\\Python Master\\Project Files\\data\\peaks\\Baseline\\Peaks_Part'+str(part)+".csv"))
                            scr = (len(base_peak) / float(len(frame)/sampleRate)) * 60
                        HR_avg = frame["HR"].mean()
                        HR_max = []
                        for j in range(0, int(len(frame)/tenSec)+1):
                            last = j*tenSec+tenSec
                            if last <=len(frame):
                                loc = frame.iloc[j*tenSec:last]
                            else:
                                loc = frame.iloc[j*tenSec:len(frame)]
                            HR_max.append(loc.HR.max())
                        HR_max = sum(HR_max)/len(HR_max)
                        scl = frame['filtered_eda'].mean()                       
                        var_list.append([part, video, condition, seg, HR_avg, HR_max,scr,scl])                
                    seg += 1
            except IndexError:
                print("Participant", part, "failed")
    files.save(var_list, headers, "Participant", "Segments")
    files.save(var_list, headers, "Participant", "data\\"+study+"\\Segments")
            
def apply_tags(data, condition, tags, start = None):
    n = 0
    events = None
    
    if condition == "park":
        start_video = start
        local_tags = tags.copy(deep=True)
        n+=1
        local_index = []            
        for j in range(0,len(local_tags)):
            local_index.append(start_video + local_tags.index[j])           
        local_tags.index = local_index
        if events is None:
            events = local_tags
        else: 
            events = pd.concat([events, local_tags])
    else:
        videos = VIDEOS[condition]
        for i in range(0,len(data)):
            if data["Event"].iloc[i] == 1:
                
                local_tags = tags[str(videos[n])].copy(deep = True)
                n = n + 1
                
                # transform times in tags
                start_video = data["Event"].index[i]
                local_index = []            
                for j in range(0,len(local_tags)):
                    local_index.append(start_video + local_tags.index[j])           
                local_tags.index = local_index
                if events is None:
                    events = local_tags
                else: 
                    events = pd.concat([events, local_tags])
    
    data = pd.concat([data, events])
    data = data.sort_index()     
    return data
                                  
def snipping(data, snipping):
    snipping = dt.timedelta(seconds = snipping/2)
    index_by_tag = {}
    time_of_tag = {}
    
    for i in range(0,len(data)):
        if pd.notna(data["Tag"].iloc[i]):
            tag_name = data["Tag"].iloc[i]
            tag_time = data.index[i]
            mask = (data.index > tag_time - snipping) & (data.index <= tag_time + snipping)
            tag_index = data.loc[mask].index
            if tag_name not in index_by_tag:
                index_by_tag[tag_name] = []
                time_of_tag[tag_name] = []
            
            index_by_tag[tag_name].append(tag_index)
            time_of_tag[tag_name].append(tag_time)
    
    return index_by_tag, time_of_tag
                        
def prepare_tags(filepath):#integrate save tags if not existant, move to svm.py
    base = files.load(os.path.abspath(filepath+'\\Segments.csv'))
    filelist = os.listdir(os.path.abspath(filepath+'\\clean'))
    for file in filelist:
        proc_data = {}
        if file.startswith("Participant"):
            data = files.load(os.path.abspath(filepath+'\\clean\\'+file))
            snip_index, snip_time = snipping(data, snip_len)
            part = int(re.search('Participant(.+?)_',file).group(1))
            SCR = pd.io.parsers.read_csv(os.path.abspath(filepath+'\\peaks\Peaks_Part'+str(part)+".csv"), dtype = {'EDA': np.float64}, index_col = [0], header = [0], parse_dates = True)#mach des wieder load
            SCR_max = SCR.EDA.max()
            baseline = base[(base["Participant"]==part)&(base["Video"]==0)]
            for tag in snip_index:
                case = snip_index[tag]
                #case_time = snip_time[tag]
                for i in range(0,len(case)): #wo anders hi
                    case_index = case[i]
                    case_data = data.loc[case_index]
                    case_SCR = SCR[(SCR.index >= case_index[0])&(SCR.index <= case_index[-1])]
                    
                    #HR
                    HR_avg = check_finite((case_data["HR"].mean()-baseline["HR_avg"].item())/(lim_HR-baseline["HR_avg"].item()))
                    HR_max = check_finite((case_data["HR"].max()-baseline["HR_avg"].item())/(lim_HR-baseline["HR_avg"].item()))
                    HR_minmax = check_finite((case_data["HR"].max()-case_data["HR"].min())/(lim_HR-baseline["HR_avg"].item()))
                    
                    #HRV
                    NN = case_data[(case_data["IBI"]>-1)&(case_data["IBI"]<2000)]
                    NN_avg = check_finite((NN.IBI.mean())/(snip_len*1000))
                    SDNN = check_finite((NN.IBI.std())/(snip_len*1000))
                    
                    NN = NN.IBI.tolist()
                    NNdif = []
                    for i in range(0, len(NN)-1):
                        NNdif.append(abs(NN[i]-NN[i+1]))
                    NNdif = np.asarray(NNdif)
                    SDSD = check_finite((NNdif.std())/(snip_len*1000))
                    RMSSD = check_finite(np.sqrt(np.mean(NNdif**2))/((snip_len*1000)))
                    try:
                        pNN20 = check_finite(float(len([x for x in NNdif if (x>20)])) / float(len(NNdif)))
                    except ZeroDivisionError:
                        pNN20 = 0
                    try:
                        pNN50 = check_finite(float(len([x for x in NNdif if (x>50)])) / float(len(NNdif)))
                    except ZeroDivisionError:
                        pNN50 = 0
                    
                    #EDA
                    SCR_avg = check_finite(case_SCR.EDA.mean()/SCR_max)
                    SCR_n = check_finite((len(case_SCR)/snip_len)*60)
                                    
                    variables = [HR_avg, HR_max, HR_minmax, NN_avg, SDNN, SDSD, RMSSD, pNN20, pNN50, SCR_avg, SCR_n]
                    try:
                        proc_data[tag].append(variables)
                    except KeyError:
                        proc_data[tag] = [variables]
                
            files.save_dict(proc_data,"data\\vr\\Tags_Participant_"+str(part))#save to proper place

def check_finite(x):
    if np.isfinite(x) and x > 0.00001:
        return x
    else:
        return 0
            
def load_for_classification(filepath, events, included_variables): #make adaptable
    data = []
    cat = []
    filelist = os.listdir(filepath)
    for file in filelist:
        if file.startswith("Tags_"):
            data_dic = files.load_dict(os.path.join(filepath,file))
            
            for tag in data_dic:
                if tag == "Start" or tag == "End" or tag == "End brake" or events[tag] == False:
                    pass
                else:
                    for data_point in data_dic[tag]:
                        for i in range(0,len(data_point)):
                            if abs(data_point[i]) < 0.0001 and data_point[i] != 0:
                                data_point[i] = 0
                            try:
                                x = data_point[i]
                                x += 1
                            except:
                                print(data_point)
                        data_point = list(map(float, data_point))
                        data.append(data_point)
                        cat.append(TAGS[tag])
                        
    included = []
    for tag in included_variables:
        if included_variables[tag] == True:
            included.append(tag)
                        
    columns = ["HR_avg", "HR_max", "HR_minmax", "NN_avg", "SDNN", "SDSD", "RMSSD", "pNN20", "pNN50", "SCR_avg", "SCR_n"]
    data = pd.DataFrame(np.asarray(data), columns = columns)
    data = data[included]
    for i,j in zip(*np.where(pd.isnull(data))):
        data.iloc[i,j] = 0
    data.to_csv("blub.csv")
    cat = pd.DataFrame(np.asarray(cat), columns = ["cat"])
    return data, cat

def part_data(filepath):
    filepath = os.path.join(filepath, "raw")
    filelist = os.listdir(filepath)
    start = []
    end = []
    j=0
    
    for i in range(0,len(COASTER)):
        srt = pd.to_datetime(COASTER[i], unit='s')
        start.append(srt)
        end.append(srt + pd.to_timedelta(DUR[i]+60, unit='s'))
    
    data, sampleRate = files.Acceleration(r"C:\Users\louwa\Documents\Python Master\Project Files\data\park\static\Motion.csv")
    for i in range(0, len(start)):
        mask = (data.index >= start[i]) & (data.index <= end[i])
        seg = data.loc[mask]
        seg.to_csv("data\\park\\clean\\Coaster"+str(i+1)+"\\Motion.csv")
    
    for file in filelist:
        start = []
        end = []
        if file.startswith("1561"):
            part = re.search('_(.+?)$',file).group(1)
            j+=1
            datapath = os.path.join(filepath,file)
            marker = pd.read_csv(os.path.join(datapath,"tags.csv"), header = None)
            for i in range(0,len(marker)):
                srt = pd.to_datetime(marker.iloc[i].item(), unit = 's')-pd.to_timedelta(20, unit='s')
                for k in range(i,len(COASTER)):
                    if abs(pd.to_datetime(COASTER[k], unit='s')-srt) < pd.to_timedelta(300, unit = 's'):
                        start.extend([None]*(k-len(start)))
                        start.append(srt)
                        end.extend([None]*(k-len(end)))
                        end.append(srt + pd.to_timedelta(DUR[k]+60, unit='s'))
                    
            data, sampleRate = files.loadData_E4(datapath)
            for i in range(0, len(start)):
                if start[i] != None:
                    mask = (data.index >= start[i]) & (data.index <= end[i])
                    seg = data.loc[mask]
                    if len(seg)>0:
                        seg.to_csv("data\\park\\clean\\Coaster"+str(i+1)+"\\Part"+part+"_Id"+str(j)+".csv")
                else:
                    pass
                
            seg = data.iloc[0:10*60*sampleRate]
            seg.to_csv("data\\park\\clean\\Baseline\\Part"+part+"_Id"+str(j)+".csv")
            
def segment_e4(filepath, sampleRate, segments):
    headers = ["Participant","Video","Condition","Segment","HR_avg","HR_max","SCR","SCL"]
    var_list = []
    
    for folder in os.listdir(filepath):
        folderpath = os.path.join(filepath,folder)
        for file in os.listdir(folderpath):
            if file.startswith("Part"):
                data = files.load(os.path.join(folderpath,file))
                condition = 0                    
                part = int(re.search('Id(.+?)\.csv',file).group(1))
                peak_data = files.load(os.path.abspath(r"C:\\Users\\louwa\\Documents\\Python Master\\Project Files\\data\\park\\peaks\\"+folder+"\\Peaks_Part"+str(part)+".csv"))                
        
                if folder == "Baseline":
                    seg = 0
                    video = 0
                    frame = data
                    scr_peak = peak_data
                    HR_avg, HR_max,scr,scl = fetch_var(frame, scr_peak,sampleRate)
                    var_list.append([part, video, condition, seg, HR_avg, HR_max,scr,scl])
                else:
                    seg = 1
                    video = int(folder[-1])
                    if video == 7:
                        video = 6
                    for i in segments:
                        try:                        
                            start = data[data["Tag"]==i[0]].iloc[0].name
                            try:
                                start = data.index.get_loc(start).start
                            except AttributeError:
                                start = data.index.get_loc(start)
                            end = data[data["Tag"]==i[1]].iloc[0].name
                            try:
                                end = data.index.get_loc(end).start
                            except AttributeError:
                                end = data.index.get_loc(end)
                            
                            print(file, folder)
                            frame = data.iloc[start:end]
                            starttime = data.iloc[start].name
                            endtime = data.iloc[end].name 
                            mask = (peak_data.index >= starttime) & (peak_data.index <= endtime)
                            scr_peak = peak_data.loc[mask]
                            HR_avg, HR_max,scr,scl = fetch_var(frame, scr_peak, sampleRate)
                            var_list.append([part, video, condition, seg, HR_avg, HR_max,scr,scl])
                            seg+=1
                        except IndexError:
                            print(i, part, video)
    
    files.save(var_list, headers, "Participant", "Segments")
    files.save(var_list, headers, "Participant", "data\\park\\Segments")

def fetch_var(frame, scr_peak, sampleRate):
    HR_avg = frame["HR"].mean()
    HR_max = frame["HR"].max()
    scl = frame['filtered_eda'].mean()
    scr = (len(scr_peak) / float(len(frame)/sampleRate)) * 60
    
    return HR_avg, HR_max,scr,scl                 

def prepare_tags_e4(filepath):#integrate save tags if not existant, move to svm.py
    base = files.load('data\\Segments.csv')
    for folder in os.listdir(filepath):
        if folder.startswith("Coaster"):
            folderpath = os.path.join(filepath, folder)
            filelist = os.listdir(folderpath)
            for file in filelist:
                proc_data = {}
                if file.startswith("Part"):
                    data = files.load(os.path.join(folderpath,file))
                    snip_index, snip_time = snipping(data, snip_len)
                    part = int(re.search('Id(.+?)\.csv',file).group(1))
                    SCR = pd.io.parsers.read_csv(os.path.abspath('C:\\Users\\louwa\\Documents\\Python Master\\Project Files\\data\\peaks\\'+folder+'\\Peaks_Part'+str(part)+".csv"), dtype = {'EDA': np.float64}, index_col = [0], header = [0], parse_dates = True)#mach des wieder load
                    SCR_max = SCR.EDA.max()
                    baseline = base[(base["Participant"]==part)&(base["Video"]==0)]
                    for tag in snip_index:
                        case = snip_index[tag]
                        #case_time = snip_time[tag]
                        for i in range(0,len(case)): #wo anders hi
                            case_index = case[i]
                            case_data = data.loc[case_index]
                            case_SCR = SCR[(SCR.index >= case_index[0])&(SCR.index <= case_index[-1])]
                            
                            #HR
                            HR_avg = check_finite((case_data["HR"].mean()-baseline["HR_avg"].item())/(lim_HR-baseline["HR_avg"].item()))
                            HR_max = check_finite((case_data["HR"].max()-baseline["HR_avg"].item())/(lim_HR-baseline["HR_avg"].item()))
                            HR_minmax = check_finite((case_data["HR"].max()-case_data["HR"].min())/(lim_HR-baseline["HR_avg"].item()))
                            
                            #HRV
                            NN = case_data[(case_data["IBI"]>-1)&(case_data["IBI"]<2000)]
                            NN_avg = check_finite((NN.IBI.mean())/(snip_len*1000))
                            SDNN = check_finite((NN.IBI.std())/(snip_len*1000))
                            
                            NN = NN.IBI.tolist()
                            NNdif = []
                            for i in range(0, len(NN)-1):
                                NNdif.append(abs(NN[i]-NN[i+1]))
                            NNdif = np.asarray(NNdif)
                            SDSD = check_finite((NNdif.std())/(snip_len*1000))
                            RMSSD = check_finite(np.sqrt(np.mean(NNdif**2))/((snip_len*1000)**2))
                            try:
                                pNN20 = check_finite(float(len([x for x in NNdif if (x>20)])) / float(len(NNdif)))
                            except ZeroDivisionError:
                                pNN20 = 0
                            try:
                                pNN50 = check_finite(float(len([x for x in NNdif if (x>50)])) / float(len(NNdif)))
                            except ZeroDivisionError:
                                pNN50 = 0
                            
                            #EDA
                            SCR_avg = check_finite(case_SCR.EDA.mean()/SCR_max)
                            SCR_n = check_finite((len(case_SCR)/snip_len)*60)
                                            
                            variables = [HR_avg, HR_max, HR_minmax, NN_avg, SDNN, SDSD, RMSSD, pNN20, pNN50, SCR_avg, SCR_n]
                            try:
                                proc_data[tag].append(variables)
                            except KeyError:
                                proc_data[tag] = [variables]
                        
                    files.save_dict(proc_data,"data\\tags\\"+folder+"\\Tags_Participant_"+str(part))#save to proper place
    
def load_for_classification_e4(filepath, events, included_variables): #make adaptable
    data = []
    cat = []
    for folder in os.listdir(filepath):
        folderpath = os.path.join(filepath, folder)
        filelist = os.listdir(folderpath)
        for file in filelist:
            if file.startswith("Tags_"):
                data_dic = files.load_dict(os.path.join(folderpath, file))
                
                for tag in data_dic:
                    if tag == "Start" or tag == "End" or tag == "End brake" or events[tag] == False:
                        pass
                    else:
                        for data_point in data_dic[tag]:
                            for i in range(0,len(data_point)):
                                if abs(data_point[i]) < 0.0001 and data_point[i] != 0:
                                    data_point[i] = 0
                                try:
                                    x = data_point[i]
                                    x += 1
                                except:
                                    print(data_point)
                            data_point = list(map(float, data_point))
                            data.append(data_point)
                            cat.append(TAGS[tag])
                        
    included = []
    for tag in included_variables:
        if included_variables[tag] == True:
            included.append(tag)
                        
    columns = ["HR_avg", "HR_max", "HR_minmax", "NN_avg", "SDNN", "SDSD", "RMSSD", "pNN20", "pNN50", "SCR_avg", "SCR_n"]
    data = pd.DataFrame(np.asarray(data), columns = columns)
    data = data[included]
    for i,j in zip(*np.where(pd.isnull(data))):
        data.iloc[i,j] = 0
    data.to_csv("blub.csv")
    cat = pd.DataFrame(np.asarray(cat), columns = ["cat"])
    return data, cat

def frame(data, frm, to, n=0):
    srt = data[data["Tag"]==frm].iloc[n].name
    try:
        srt = data.index.get_loc(srt).start
    except AttributeError:
        srt = data.index.get_loc(srt)
    try:
        end = data[data["Tag"]==to].iloc[n].name
        try:
            end = data.index.get_loc(end).start
        except AttributeError:
            end = data.index.get_loc(end)
    except IndexError:
        end = None
    return srt, end