# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:36:13 2019

@author: Eric Lehman
"""
import pandas as pd


def get_headers(header_row):
    """ Takes array of headers and sets up a dictionary corresponding to columns. """
    ans = {}
    for i in range(len(header_row)):
        ans[header_row[i]] = i
        
    return ans

def load(f = './attn_logs_test.csv'):
    """ Loads the data, and gets a proper extraction of the header and data. """
    df = pd.read_csv(f, header = 0)
    df['attention_acceptance'].fillna("auc", inplace = True)
    loh = list(df.columns)
    header = get_headers(loh)
    return df, header

def best_val_f1():
    df, header = load()
    best_loc = df[df['data_config'] == 'vanilla'][df['cond_attn'] == False]['best_val_f1'].idxmax()
    row = df.iloc[best_loc]
    print("Article Encoder:       {}".format(row['article_encoder']))
    print("ICO Encoder:           {}".format(row['ico_encoder']))
    print("Attention Acceptance:  {}".format(row['attention_acceptance']))
    print("Use attention:         {}".format(row['attn']))
    print("Tokenwise attn:        {}".format(row['tokenwise_attention']))
    print("Tune Embeddings:       {}".format(row['tune_embeddings']))
    print("Pretrain attention:    {}".format(row['pretrain_attention']))
    print("Cond attn:             {}".format(row['cond_attn']))
    print("Best Val F1:     {:.3f}".format(row['best_val_f1']))
    print("Test F1:         {:.3f}".format(row['test_f1']))
    print("Test AUC:        {:.3f}".format(row['test_auc']))
    print("Test Precision:  {:.3f}".format(row['test_p']))
    print("Test Recall:     {:.3f}".format(row['test_r']))
    print("Test Entropy:    {:.3f}".format(row['test_entropy']))
    print("Test mass:       {:.3f}".format(row['test_evidence_token_mass']))
best_val_f1()
