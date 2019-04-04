# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 00:02:05 2018

@author: Eric
"""
import pandas as pd
import numpy as np

TEST_F1_NAME  = "test_f1"
TEST_ACC_NAME = "test_acc"
TEST_PR_NAME  = "test_p"
TEST_RC_NAME  = "test_r"
TEST_LOSS     = "val_loss"
TRAIN_LOSS    = "train_loss"
EPOCH         = "epoch" 
TOTAL_EPOCHS  = "epochs"

def gen_all_attention_type():
    data, header = load_data_all_neural()
    # find all other pretraining types 
    other_pretrain = set([t[header['pretrain_attention']] for t in data])
    other_pretrain.remove("False")
    other_pretrain.remove("pretrain_attention")
    other_pretrain = list(other_pretrain) 
    return other_pretrain

def get_row_headers(df):
    """ Get row headers. """
    col = df.columns.values
    dictionary = {}
    i = 0 
    for n in col:
        dictionary[n] = i
        i += 1

    return dictionary 

def model_key(row, headers, use_problem = True):
    """ Generate a unique key for models that make them identifiable. """
    model = ["ico_encoder", "article_encoder", "attn", "cond_attn", 
             "tokenwise_attention", "pretrain_attention", 
             "tune_embeddings", "no_pretrained_word_embeddings"]
    problem = ["article_sections", "data_config"]
    
    key = ""
    for id_ in model:
        key += row[headers[id_]] + "|||"
        
    if (use_problem):
        for id_ in problem:
            key += row[headers[id_]] + "|||"
    
    return key

def load_data_all_neural():
    # load in file
    #l = #'./data/combined5.2018-12-03.enforce_sample_ordering.csv'
    l = './data/combined6.attn_fixups.changeattnloss.2018-12-03.csv'
    # l = './data/combined5_attn_logs5_more_epochs.2018-11-30.csv'
    df = pd.read_csv(l)
    headers = get_row_headers(df)
    df.fillna("")
    return np.asarray(df), headers

def add_entry(data, model_key, row, headers):
    """ Add data from row to dictionary"""
    ep  = row[headers[EPOCH]]
    tep = row[headers[TOTAL_EPOCHS]]
    #vls = [row[headers[TEST_LOSS]], "val_loss"]
    #tls = [row[headers[TRAIN_LOSS]], "train_loss"]
    f1  = [row[headers[TEST_F1_NAME]], "f1"]
    acc = [row[headers[TEST_ACC_NAME]], "acc"]
    pc  = [row[headers[TEST_PR_NAME]], "precision"]
    rc  = [row[headers[TEST_RC_NAME]], "recall"]
    entries = [acc, pc, rc, f1]
    #loss = [vls, tls]
    
    # if this a new entry
    if not(model_key in data):
        tmp = {}
        # put all of the row into an array with easy access via dictionary
        for key in headers.keys():
            tmp[key] = row[headers[key]]
        
        data[model_key] = {"full_entry": tmp}

    for e in entries:
        if (pd.isnull(e[0]) or e[0] == ""):
            continue
        else:
            data[model_key][e[1]] = e[0]
    
    """        
    for l in loss:
        try: 
            tep = int(tep)
            ep  = int(ep)
        except:
            continue
        
        if (pd.isnull(l[0]) or l[0] == ""):
            continue
        
        if not(l[1] in data[model_key]):
            data[model_key][l[1]] = [0] * int(tep)
            
        data[model_key][l[1]][int(ep)] = l[0]
    """
    return data

def parse_all_experiments(variants):
    """ Parse all of the experiments printed out from the CSV files. """
    df, headers = load_data_all_neural()
    data = {}
    
    # iterate through and tag each row
    for row in df: 
        key = model_key(row, headers, use_problem = True)
        add_entry(data, key, row, headers)
    
    # find ones that satisfy variants (AKA filter)
    for k in {**data}.keys():
        model = data[k]
        flag  = True
        # if the specification (i.e. attn) we are looking for == the model's
        for specification in variants.keys():
            if not(model["full_entry"][specification] == variants[specification]):
                flag = False
        
        # if the variants are not satisfied... remove...
        if not(flag):
            data.pop(k, None)
        
    return data

def find_best_f1(variants):
    """ Find the best F1 performance based on the given variants. """
    d = parse_all_experiments(variants)
    idx = ""
    best_f1 = 0
    for k in d.keys():
        model_f1 = float(d[k]['f1'])
        if (model_f1 > best_f1):
            idx = k
            best_f1 = model_f1
        
    if idx == "":
        return None
    
    # best_acc, best_prc, best_rc = d[idx]["acc"], d[idx]["precision"], d[idx]['recall']
    return d[idx]    

def extract_model_configs(full_entry):
    """ Given a full entry, extract model configurations and put into a dict. """
    model = ["ico_encoder", "article_encoder", "attn", "cond_attn", 
             "tokenwise_attention", "pretrain_attention", 
             "tune_embeddings", "no_pretrained_word_embeddings"]
    problem = ["article_sections", "data_config"]
    d = {}
    for m in model:
        d[m] = full_entry[m]
        
    for p in problem:
        d[p] = full_entry[p]
        
    return d

def formalize_table_neural_experiments(name, d):
    """ Takes the data from find_best_f1 and parses it, and returns an array with name + performance. """
    if (d is None):
        return [name, 0, 0, 0, 0]
    
    best_f1, best_acc, best_prc, best_rc = d["f1"], d["acc"], d["precision"], d['recall']
    best_f1, best_acc, best_prc, best_rc = ["{:.2f}".format(float(inst)) for inst in [best_f1, best_acc, best_prc, best_rc]]
    return [name, best_f1, best_acc, best_prc, best_rc]