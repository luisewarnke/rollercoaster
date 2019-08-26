# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:04:02 2019

@author: louwa
"""

import data_prep as dp
import classification as clas
import exploration
import os

#filepath = r"C:\Users\louwa\Documents\Python Master\Project Files\data"
filepath = r"C:\Users\louwa\Documents\Python Master\Project Files\data\vr"
filelist = os.listdir(filepath)

SEG = [["base", "Start", 1],
       ["Start", "Lift hill", 3],
       ["Lift hill", "First drop", 3],
       ["First drop", "End brake", 3],
       ["End brake", "End", 3]]

SEG2 = [["Start", "Lift hill"],
       ["Lift hill", "First drop"],
       ["First drop", "End brake"],
       ["End brake", "End"]]

starts = []

def main():
    #dp.clean_raw(filepath)
    dp.segment(filepath, 256, SEG, "vr") #to exploration
    #exploration.test_segments(filepath)
    #dp.prepare_tags(filepath)
    #clas.analyse()
    
#    dp.part_data(filepath)
#    raw_files = os.path.join(filepath,"clean")
#    for file in os.listdir(raw_files):
#        dp.clean_raw(raw_files, False, file)
#    dp.segment_e4(raw_files,8, SEG2)
#    exploration.test_segments(filepath)
#    dp.prepare_tags_e4(filepath)
    
    
main()