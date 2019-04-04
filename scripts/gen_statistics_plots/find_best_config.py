# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 23:02:24 2019

@author: Eric Lehman
"""

from load_data import load, filter_data

def find_best(restraints):
    """ Find the best config.  """
    # load data and get ready to search 
    df, header = load()
    filt  = filter_data(df, header, restraints)
    search     = 'best_val_f1'; loc = header[search]
    
    # search for max
    m    = 0
    best = None
    for f in filt:
        if f[-1][loc] > m:
            m = f[-1][loc] 
            best = f[-1]
            
    return gen_config(restraints, header, best)

def gen_config(restraints, header, best):
    if best is None:
        raise Exception("No rows with the given requirements exist.")
    config = ['article_sections', 'ico_encoder', 'article_encoder', 'attn', 'cond_attn', 'tokenwise_attention', 'data_config', 'pretrain_attention']
    for c in config:
        restraints[c] = set([best[header[c]]])
            
    
    return restraints

        
