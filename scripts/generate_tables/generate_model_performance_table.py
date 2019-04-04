# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:22:01 2018

@author: Eric
"""
import sys
import torch
import random
import numpy as np
from tabulate import tabulate
sys.path.append("..//..//.//evidence_inference//models//") # Adds higher directory to python modules path.
sys.path.append("..//..//.//evidence_inference//experiments//") # Adds higher directory to python modules path.

from heuristics import main as run_hr
from regression import main as run_LR
from heuristics_cheating import main as run_hr_cheating
from regression_cheating import main as run_LR_cheating
from table_parse_helper import find_best_f1, extract_model_configs, formalize_table_neural_experiments, gen_all_attention_type


def get_LR():
    """ Returns the LR and Guessing results in a formatted row fashion. """
    np.random.seed(500)
    random.seed(500)
    torch.manual_seed(500)

    i = 100
    acc, f1, prec, rec, acc_g, f1_g, prec_g, rec_g = run_LR(i, True)
    LR = ["Logistic Regression", "{:.2f}".format(acc), "{:.2f}".format(prec), "{:.2f}".format(rec), "{:.2f}".format(f1)]
    GE = ["Majority Guessing", "{:.2f}".format(acc_g), "{:.2f}".format(prec_g), "{:.2f}".format(rec_g), "{:.2f}".format(f1_g)]
    return [LR, GE]

def get_LR_cheating():
    """ Grab the heuristic cheating results. """ 
    np.random.seed(500)
    random.seed(500)
    torch.manual_seed(500)

    i = 100
    acc, f1, prec, rec, _, _, _, _ = run_LR_cheating(i, True)
    LR = ["Logistic Regression Cheating", "{:.2f}".format(acc), "{:.2f}".format(prec), "{:.2f}".format(rec), "{:.2f}".format(f1)]
    return LR

def get_heuristics():
    """ Returns the heuristic results. """
    acc, f1, prec, rec = run_hr()
    hr = ["Heuristics", "{:.2f}".format(acc), "{:.2f}".format(prec), "{:.2f}".format(rec), "{:.2f}".format(f1)]
    return hr

def get_heuristics_cheating():
    """ Grab the heuristic cheating results. """
    acc, f1, prec, rec = run_hr_cheating()
    hr = ["Heuristics", "{:.2f}".format(acc), "{:.2f}".format(prec), "{:.2f}".format(rec), "{:.2f}".format(f1)]
    return hr

def table_2_neural_experiments():
    """ The function for getting the best model results."""
    # get configuration for best attention/no-attention + vanilla
    attn_v    = find_best_f1({"attn": 'True', 'data_config': 'vanilla'})
    no_attn_v = find_best_f1({"attn": 'False', 'data_config': 'vanilla'})
    
    #print(extract_model_configs(attn_v['full_entry']))
    # Best same-config model but flipped attention attention
    config_attn_v    = extract_model_configs(attn_v['full_entry'])
    config_no_attn_v = extract_model_configs(no_attn_v['full_entry'])
    config_attn_v['attn']    = 'False'
    config_no_attn_v['attn'] = 'True'

    all_attn = ["pretrain_attention", "tokenwise_attention"]
    for tp in all_attn:
        if tp in config_attn_v:
            del config_attn_v[tp]
        if tp in config_no_attn_v:
            del config_no_attn_v[tp]
    
    cmp_attn_v    = find_best_f1(config_attn_v)
    cmp_no_attn_v = find_best_f1(config_no_attn_v)
    
    ret = [["Best model vanilla attn [1]", attn_v], 
           ["Best model vanilla w/o attn [2]", no_attn_v], 
           ["Same config as model [1] but w/o attn [3]", cmp_attn_v],
           ["Same config as model [2] but w/ attn [4]", cmp_no_attn_v]]
    
    data = [formalize_table_neural_experiments(inst[0], inst[1]) for inst in ret]
    #attn_c    = find_best_f1({"attn": 'True', 'data_config': 'cheating'})
    #no_attn_c = find_best_f1({"attn": 'False', 'data_config': 'cheating'})
    
    return data
    
    
def gen_table_data():
    """ Generate a table. """
    c_header = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    table    = [c_header]
    hr       = get_heuristics()
    hr_cheat = get_heuristics_cheating()
    LR, GE   = get_LR()
    LR_cheat = get_LR_cheating()
    #all_exp  = table_2_neural_experiments()
    
    table.append(GE)
    table.append(hr)
    table.append(hr_cheat)
    table.append(LR)
    table.append(LR_cheat)
    """
    for row in all_exp:
        table.append(row)
    """
    return table

def gen_latex_table(table):
    """ Generate latex table based on table. """ 
    content = "\\begin{table*}\n"
    content += tabulate(table[1:], headers = table[0], tablefmt="latex")
    content += "\n\\end{table*}"
    return content


if __name__ == "__main__":
    table = gen_table_data()
    latex = gen_latex_table(table)
    print(latex)
