# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:08:48 2019

@author: Eric Lehman

This file is for producing the table for the validation results (NN).
"""
import random
import pandas as pd
from tabulate import tabulate

def fmt(avg_, min_, max_, n):
    return "{:.3f} ({:.3f}, {:.3f})".format(avg_[n], min_[n], max_[n])

def reg_fmt(avg_, n):
    return "{:.3f}".format(avg_[n])

def sample(df, n = 5):
    if len(df) == 0:
        raise Exception("DF is empty.")
        
    num = list(range(len(df))); random.shuffle(num); to_keep = num[:n]
    max_, min_ = df.iloc[to_keep[0]], df.iloc[to_keep[0]]
    f1, prec, recall, token_auc, mass = 0, 0, 0, 0, 0
    for k in to_keep:
        row = df.iloc[k]
        f1 += row['test_f1'] / n
        prec += row['test_p'] / n
        recall += row['test_p'] / n
        token_auc += row['test_auc'] / n
        mass += row['test_evidence_token_mass'] / n
     
        if row['test_f1'] > max_['test_f1']:
            max_ = row
            
        if row['test_f1'] < min_['test_f1']:
            min_ = row
            
    return max_, min_, {'test_f1': f1, 'test_p': prec, 'test_r': recall, 'test_auc': token_auc, 'test_evidence_token_mass': mass}

def gen_latex_table(table, names):
    """ Generate latex table based on table. """ 
    table = table_preprocess(table, names)
    content = "\\begin{table*}\n"
    content += tabulate(table[1:], headers = table[0], tablefmt="latex")
    content += "\n\\end{table*}"
    return content


def table_preprocess(table, names):
    new_table = [['Model', 'Precision', 'Recall', 'F1', 'Token AUC / Mass']]
    def one_row(max_, min_, avg_, n):
        return [n, fmt(avg_, min_, max_, 'test_p'), 
                fmt(avg_, min_, max_, 'test_r'),
                fmt(avg_, min_, max_, 'test_f1'), 
                "{} / {}".format(reg_fmt(avg_, 'test_auc'), reg_fmt(avg_, 'test_evidence_token_mass'))]
    
    i = 0
    for row in table:
        max_, min_, avg_ = row
        new_table.append(one_row(max_, min_, avg_, names[i]))
        
        i += 1

    return new_table

def get_headers(header_row):
    """ Takes array of headers and sets up a dictionary corresponding to columns. """
    ans = {}
    for i in range(len(header_row)):
        ans[header_row[i]] = i
        
    return ans

def load(f = './attn_logs_val.csv'):
    """ Loads the data, and gets a proper extraction of the header and data. """
    df = pd.read_csv(f, header = 0)
    df['attention_acceptance'].fillna("auc", inplace = True)
    loh = list(df.columns)
    header = get_headers(loh)
    return df, header

def get_results(df, restraints):
    new_df = df
    for key in restraints.keys():

        new_df = new_df[new_df[key] == restraints[key]]
        
        
        
    val_f1_rows = new_df[pd.notnull(new_df['best_val_f1'])]
    l_max, l_min, avg = sample(val_f1_rows)

    
    return l_max, l_min, avg


# Mix with 3 styles of attention acceptance
attn1     = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'auc', 'data_config': 'vanilla', 'pretrain_attention': 'False'}

pre_attn1 = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'auc', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention'}
pre_attn2 = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'entropy', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention'}
pre_attn3 = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'evidence_mass', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention'}

pre_attn4 = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'auc', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention_balanced'}
pre_attn5 = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'entropy', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention_balanced'}
pre_attn6 = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'evidence_mass', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention_balanced'}

pre_attn7 = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'auc', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_max_evidence_attention'}
pre_attn8 = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'entropy', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_max_evidence_attention'}
pre_attn9 = {'attn': True, 'cond_attn': False, 'attention_acceptance': 'evidence_mass', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_max_evidence_attention'}

cond_attn1 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'auc', 'data_config': 'vanilla', 'pretrain_attention': 'False'}


pre_cond_attn1 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'auc', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention'}
pre_cond_attn2 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'entropy', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention'}
pre_cond_attn3 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'evidence_mass', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention'}

pre_cond_attn4 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'auc', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention_balanced'}
pre_cond_attn5 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'entropy', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention_balanced'}
pre_cond_attn6 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'evidence_mass', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_tokenwise_attention_balanced'}

pre_cond_attn7 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'auc', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_max_evidence_attention'}
pre_cond_attn8 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'entropy', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_max_evidence_attention'}
pre_cond_attn9 = {'attn': True, 'cond_attn': True, 'attention_acceptance': 'evidence_mass', 'data_config': 'vanilla', 'pretrain_attention': 'pretrain_max_evidence_attention'}

config = [attn1, 
          pre_attn1, pre_attn2, pre_attn3, pre_attn4, pre_attn5, pre_attn6, pre_attn7, pre_attn8, pre_attn9,
          cond_attn1,
          pre_cond_attn1, pre_cond_attn2, pre_cond_attn3, pre_cond_attn4, pre_cond_attn5, pre_cond_attn6, pre_cond_attn7, pre_cond_attn8, pre_cond_attn9]


df, _ = load()
table = []
for c in config:
    df_row = get_results(df, c)
    table.append(df_row)    
    
names = ['+ Attn.', 
         '+ Pretrain attn. [AUC] (Tokenwise attn.)', 
         '+ Pretrain attn. [Entropy] (Tokenwise attn)', 
         '+ Pretrain attn. [Evidence Mass] (Tokenwise attention)',
         '+ Pretrain attn. [AUC] (Tokenwise attn. balanced)', 
         '+ Pretrain attn. [Entropy] (Tokenwise attn. balanced)', 
         '+ Pretrain attn. [Evidence Mass] (Tokenwise attn. balanced)',
         '+ Pretrain attn. [AUC] (Max evidence attn.)', 
         '+ Pretrain attn. [Entropy] (Max evidence attn.)', 
         '+ Pretrain attn. [Evidence Mass] (Max evidence attn.)',
         '+ Cond. attn.', 
         '+ Pretrain cond. attn. [AUC] (Tokenwise attn.)', 
         '+ Pretrain cond. attention [Entropy] (Tokenwise attn.)', 
         '+ Pretrain cond. attn. [Evidence Mass] (Tokenwise attn.)',
         '+ Pretrain cond. attn. [AUC] (Tokenwise attn. balanced)', 
         '+ Pretrain cond. attn. [Entropy] (Tokenwise attn. balanced)', 
         '+ Pretrain cond. attn. [Evidence Mass] (Tokenwise attn. balanced)', 
         '+ Pretrain cond. attn. [AUC] (Max evidence attn.)', 
         '+ Pretrain cond. attn. [Entropy] (Max evidence attn.)', 
         '+ Pretrain cond. attn. [Evidence Mass] (Max evidence attn.)']

    
print(gen_latex_table(table, names))
    



