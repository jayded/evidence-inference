# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:39:22 2019

@author: Eric Lehman
"""


import sys
import numpy as np
from os.path import join, dirname, abspath


# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

from evidence_inference.preprocess import preprocessor

def calculate_entropy(Xy):
    """
    For all documents, average log(k) where k is the number of tokens in the document, 
    weighted by number of prompts for that document. 
    """
    tokens  = {} # Map article ids to token
    prompts = {} # Map article ids to num prompts
    
    for d in Xy:
        n_tokens    = len(d['article'])
        tokens[d['a_id']] = n_tokens 
        
        if d['a_id'] in prompts:
            prompts[d['a_id']] += 1
        else:
            prompts[d['a_id']] = 1
    
    total_entropy = 0
    for art in prompts.keys():
        total_entropy += np.log(tokens[art]) * prompts[art] / len(Xy)
    
    return total_entropy

tr_ids, val_ids, te_ids = preprocessor.train_document_ids(), preprocessor.validation_document_ids(), preprocessor.test_document_ids()
train_Xy, inference_vectorizer = preprocessor.get_train_Xy(tr_ids)
val_Xy  = preprocessor.get_Xy(val_ids, inference_vectorizer)
test_Xy = preprocessor.get_Xy(te_ids, inference_vectorizer)

print(calculate_entropy(train_Xy))
print(calculate_entropy(val_Xy))
print(calculate_entropy(test_Xy))

