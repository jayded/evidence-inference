# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:54:35 2018

@author: Eric

The purpose of this program is to split train and test files.
"""
import numpy as np
import pandas as pd

def load_test_docs(type_doc):
    """ Load the document ids for the test files (returns a set). """
    loc = "..//.//annotations//splits//" + type_doc + "_article_ids.txt"
    id_ = np.loadtxt(loc, delimiter = " ", dtype = int)
    return id_


def load_data(loc):
    """ Load in the csv file """
    df = pd.read_csv(loc, engine = "python", encoding = "utf-8")
    df.fillna("")
    df = np.asarray(df)
    return df

def save_data(data, loc, header):
    """ Save the data to the given location. """
    df = pd.DataFrame(data=data, columns=header)
    df.fillna("")
    df.to_csv(loc, index = False, encoding = 'utf-8')
    return None

def remove_test_file(loc, save_loc, x, header, type_doc):
    """ 
    Remove the test rows from the data. 
    
    @param loc is the location of the file.
    @param save_loc is where the file is saved
    @param x   is the location in a row where the document id is located.
    @param header is the header to use when saving the file.
    @param type_doc is one of train/test/val.
    
    """
    df = load_data(loc)
    test_ids = load_test_docs(type_doc)
    df = list(filter(lambda row: not(row[x] in test_ids), df))
    save_data(df, save_loc, header)
    return None

def main(type_doc):
    loc = '.././annotations/'
    st_a = loc + 'annotations_'
    st_p = loc + 'prompts_'
    ha = ["UserID", "PromptID", "PMCID", "Valid Label", "Valid Reasoning", 
              "Label", "Annotations", "Label Code", "In Abstract", 
              "Evidence Start", "Evidence End"]
    hp = ["PromptID", "PMCID", "Outcome", "Intervention", "Comparator"]


    to_do = [[st_a + 'merged.csv', loc + type_doc + '_annotations_merged.csv', 2, ha], 
             [st_a + 'doctor_generated.csv', loc + type_doc + '_annotations_doctor_generated.csv', 2, ha], 
             [st_a + 'pilot_run.csv', loc + type_doc + '_annotations_pilot_run.csv', 2, ha],
             [st_p + 'merged.csv', loc + type_doc + '_prompts_merged.csv', 1, hp], 
             [st_p + 'doctor_generated.csv', loc + type_doc + '_prompts_doctor_generated.csv', 1, hp], 
             [st_p + 'pilot_run.csv', loc + type_doc + '_prompts_pilot_run.csv', 1, hp]]
    
    for row in to_do:
        remove_test_file(row[0], row[1], row[2], row[3], type_doc)
   
if __name__ == '__main__':
    types = ['test', 'train', 'validation']
    for t in types:
        main(t)