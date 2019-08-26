# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:53:26 2019

@author: louwa
"""

import file_manipulation as files
import data_prep as dp
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import Imputer

filepath = r"C:\Users\louwa\Documents\Python Master\Project Files\data"
#filelist = os.listdir(filepath)

EVENTS = {"Lift hill" : True,
          "First drop" : True, 
          "Drop" : True,
          "Brake" : True,
          "Hop" : True, 
          "Tunnel" : True, 
          "Helix" : True, 
          "Loop" : True, 
          "Immelmann" : True,
          "Corkscrew" : True, 
          "Roll" : True}

VARIABLES = {"HR_avg" : True,
             "HR_max" : True,
             "HR_minmax": True,
             "NN_avg" : True,
             "SDNN":False,
             "SDSD":False,
             "RMSSD":False,
             "pNN20":True,
             "pNN50":True,
             "SCR_avg":True, 
             "SCR_n":False}

def analyse(event = False, variable = True, merge = [], method = "e4", kernel = "linear"): #cross, e4, shimmer
    if kernel != "linear":
        variable = False
    if method == "cross":
        X_train, y_train = dp.load_for_classification(os.path.abspath(filepath+"\\vr\\tags"), EVENTS, VARIABLES)
        X_test, y_test = dp.load_for_classification_e4(os.path.abspath(filepath+"\\park\\tags"), EVENTS, VARIABLES)
        
    else:   
        if method == "e4":
            data, cat = dp.load_for_classification_e4(os.path.abspath(filepath+"\\park\\tags"), EVENTS, VARIABLES)
        else:
            data, cat = dp.load_for_classification(os.path.abspath(filepath+"\\vr\\tags"), EVENTS, VARIABLES)
        X_train, X_test, y_train, y_test = train_test_split(data, cat, test_size=0.3,random_state=109)
        print(len(X_train)/(len(X_test)+len(X_train)))
    for i in merge:
        cat = cat.replace(i, i[0])

    
    clf = svm.SVC(probability = True, decision_function_shape = 'ovo', kernel = kernel)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
    if event == True:
        print(metrics.classification_report(y_test,y_pred))
    if variable == True:
        coeff = pd.DataFrame(clf.coef_).transpose(copy = True)
        for col in coeff.columns:
            coeff[col] = coeff[col]**2
        coeff["magnitude"] = coeff.sum(axis = 1)**0.5
        print(coeff["magnitude"])

def plot_data():
    meas = 0
    hr = []
    eda = []
    cat = [] #brauch i so noch nrn?
    for file in filelist:
        if file.startswith("Tags_"):
            data_dic = files.load_dict(file)

            for tag in data_dic:
                for data_point in data_dic[tag]:
                    for dat in data_point:
                        if meas == 0:
                            eda.append(dat)
                            meas = 1
                        elif meas == 1:
                            hr.append(dat)
                            meas = 0
                    cat.append(tag)
                
    cat = pd.DataFrame(np.asarray(cat),columns = ["cat"])
    hr = pd.DataFrame(np.asarray(hr),columns = ["y","x"]).join(cat)
    eda = pd.DataFrame(np.asarray(eda),columns = ["y","x"]).join(cat)
    
    sns.scatterplot("x","y","cat",data=hr)
    plt.show()
    
    sns.scatterplot("x","y","cat",data=eda)
    plt.show()