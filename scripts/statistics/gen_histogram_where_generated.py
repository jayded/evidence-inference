# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:29:14 2018

@author: Eric
"""
import glob
import string
import numpy as np
import pandas as pd 

min_occurance = 10

"""
Remove punctuation and numbers.
"""
def transform(t):
    if ("b'" in t):
        t = t.replace("b'", "").replace("'", "")
    table = str.maketrans({key: None for key in string.punctuation})
    return ''.join([i for i in t if not i.isdigit()]).translate(table).strip(" ").lower()


d, data = {}, {}
files = glob.glob(".//additional_files//*.csv")

for f in files:
    arr = np.asarray(pd.read_csv(f, engine = 'python').fillna(""))
    for row in arr:
        l = transform(row[-1])
        if (l in d):
            d[l] += 1
        else:
            d[l] = 1 

for k in d.keys(): 
    if d[k] > min_occurance:
        data[k] = d[k]
    
df = pd.DataFrame.from_dict(data, orient='index')
df.plot(kind = 'bar')