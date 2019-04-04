import torch
import argparse
import os
import sys
import random
import copy

from os.path import join, dirname, abspath

mp0 = os.path.abspath(os.path.join('../.././evidence_inference/'))
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..'))) 
from evidence_inference.preprocess import preprocessor
USE_CUDA = True
USE_TEST = True

from evidence_inference.models.scan_regression import train_scan
from evidence_inference.models.scan_regression import train_reformat, scan_reform
print("Modules loaded.")
parent_path = abspath(os.path.join(dirname(abspath(__file__)), '..', '..'))
#vocab_f = os.path.join(parent_path, "annotations", "vocab.txt")
    
def run_scan_net_regression(loc = './scan_net.pth'):
    train_Xy, inference_vectorizer = preprocessor.get_train_Xy(set(list(preprocessor.train_document_ids())), sections_of_interest=None, vocabulary_file=None, include_sentence_span_splits = True)
        
    if not(USE_TEST):
        # create an internal validation set from the training data; use 90% for training and 10% for validation.
        split_index = int(len(train_Xy) * .9)
        val_Xy = train_Xy[split_index:]
        train_Xy = train_Xy[:split_index]
        test_Xy = preprocessor.get_Xy(preprocessor.validation_document_ids(), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
    else:
        val_Xy = preprocessor.get_Xy(preprocessor.validation_document_ids(), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
        test_Xy = preprocessor.get_Xy(preprocessor.test_document_ids(), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
        
    train_Xy, val_Xy, test_Xy = train_reformat(train_Xy), scan_reform(val_Xy), scan_reform(test_Xy) 
    

    
    # train with 50 epochs, batch_size of 1, and patience of 3 (early stopping)
    model = train_scan(inference_vectorizer, train_Xy, val_Xy, test_Xy, 100, 32, 5)
    
    torch.save(model.state_dict(), loc)
