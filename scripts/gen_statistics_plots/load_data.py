# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:43:37 2019

@author: Eric Lehman
"""
import numpy as np
import pandas as pd

def get_headers(header_row):
    """ Takes array of headers and sets up a dictionary corresponding to columns. """
    ans = {}
    for i in range(len(header_row)):
        ans[header_row[i]] = i
        
    return ans

def split_on_model(df, header):
    """ Split the sections into arrays based on the model type. """
    res    = []
    tmp = [] 
    last_row = None
    for row in df:
        if len(tmp) != 0 and last_row[header['epoch']] == 'break':
            res.append(tmp)
            tmp = []
        else:
            tmp.append(row)
            last_row = row
            
    return res
    
def filter_data(df, header, restraints = {}):
    """ Filters the data based off of the restraints given, and formats it appropriately. """
    filtered = []
    for row in df: 
        to_add = True
       
        for r in restraints.keys():
            idx = header[r]
            val = row[-1][idx] # take the last row
            to_add = to_add and (val in restraints[r]) 
            
        if to_add:
           filtered.append(row) 
           
    return filtered

def extract_plotable_data(filtered, header, x_axis, y_axis):
    """ Takes filtered data, and gets the x-axis and y-axis from that row. """
    idx_x, idx_y = header[x_axis], header[y_axis]
    all_x, all_y, all_h = [], [], []
    for model in filtered:
        x, y = [], []
        all_h.append(model[0]) # Just append all the information from the first row.
        for row in model:
            try:
                x_val = float(row[idx_x])
                y_val = float(row[idx_y])
                if not(np.isnan(x_val) or np.isnan(y_val)):
                    x.append(x_val)
                    y.append(y_val)
            except:
                continue
        
        if len(x) > 0 and len(y) > 0:
            all_x.append(x)
            all_y.append(y)
    
    return all_x, all_y, all_h
            
def load(f = './attn_logs_val.csv'):
    """ Loads the data, and gets a proper extraction of the header and data. """
    df = pd.read_csv(f, header = 0)
    df['attention_acceptance'].fillna("auc", inplace = True)
    df['epoch'].fillna("break", inplace = True)


    loh = list(df.columns)
    header = get_headers(loh)
    df = np.asarray(df)
    return split_on_model(df, header), header