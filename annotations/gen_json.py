# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:32:52 2018

@author: Eric
"""
import glob
import pandas as pd

files = glob.glob("./split_files/*.csv")

for file_name in files:
    new_name = file_name.split(".csv")[0] + '.json'
    df = pd.read_csv(file_name, engine = 'python', encoding = 'utf-8')
    df.to_json(new_name)