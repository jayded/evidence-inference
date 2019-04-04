# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:24:22 2018

@author: Eric
"""
import numpy as np
import pandas as pd
from tabulate import tabulate
loc = "..\\..\\.\\annotations\\split_files\\"
train_files = [loc + "train_annotations_merged.csv", loc + "train_prompts_merged.csv"]
dev_files   = [loc + "validation_annotations_merged.csv", loc + "validation_prompts_merged.csv"]
test_files  = [loc + "test_annotations_merged.csv", loc + "test_prompts_merged.csv"]

files = [train_files, dev_files, test_files]


# #-Prompts / #-Articles / Label Breakdown

def load_data(loc):
    """ Load in CSV from given location. """
    df = pd.read_csv(loc, engine = 'python', encoding = 'utf-8')
    df.fillna("")
    return np.asarray(df)

def num_prompts(data):
    """ Find the number of unique prompts in data. """
    pmts = set()
    for row in data:
        pmts.add(row[2] + row[3] + row[4])
    return len(pmts)

def num_articles(data):
    """ Find the number of unique articles in data. """
    art = set()
    for row in data:
        art.add(row[1])
        
    return len(art)
    
def label_breakdown(data):
    """ Find the label breakdown in data. """
    res = {}
    for row in data:
        # if valid label
        if (row[3]):
            res[row[1]] = row[7]
            
    neg, neu, pos = 0, 0, 0
    for key in res.keys():
        r = res[key]
        if (r == -1):
            neg += 1
        elif (r == 0):
            neu += 1
        else: 
            pos += 1
    return "{} / {} / {}".format(neg, neu, pos)

def load_column(files):
    """ 
    Loads the results for a single column.
    
    @param files is an array of 2 locations (annotations and corresponding prompts).
    """
    # load data
    ann = load_data(files[0])
    pmt = load_data(files[1])
    
    # calculate stats
    n_pmt = num_prompts(pmt)
    n_art = num_articles(pmt)
    l_bkd = label_breakdown(ann)
    
    return [n_pmt, n_art, l_bkd]

def gen_table_data(files): 
    """ Generate a table. """
    c_header = ["", "Train Set", "Dev Set", "Test Set"]
    r_header = ["Number of Prompts", "Number of Articles", "Label Breakdown (-1/0/1)"]
    table    = [c_header]
    n_pmt, n_art, l_bkd = [r_header[0]], [r_header[1]], [r_header[2]]
    # get data for the table
    for f in files: 
        col = load_column(f)
        n_pmt.append(col[0])
        n_art.append(col[1])
        l_bkd.append(col[2])
        
    # build table
    table.append(n_pmt)
    table.append(n_art)
    table.append(l_bkd)
    
    return table

def gen_latex_table(table):
    """ Generate latex table based on table. """ 
    content = "\\begin{table*}\n"
    content += tabulate(table[1:], headers = table[0], tablefmt="latex")
    content += "\n\\end{table*}"
    return content


if __name__ == "__main__":
    table = gen_table_data(files)
    latex = gen_latex_table(table)
    print(latex)



