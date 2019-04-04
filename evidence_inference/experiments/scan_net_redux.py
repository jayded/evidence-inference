import torch
import argparse
import os
import sys
import random

from os.path import join, dirname, abspath

mp0 = os.path.abspath(os.path.join('../.././evidence_inference/'))
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..'))) 
from evidence_inference.preprocess import preprocessor
USE_CUDA = True
USE_TEST = True

use_attn = False

from evidence_inference.models.model_scan import ScanNet, train_scan
print("Modules loaded.")

def run_scan_net_redux(loc = 'scan_net_redux.pth'):
    parent_path = abspath(os.path.join(dirname(abspath(__file__)), '..', '..'))
    vocab_f = os.path.join(parent_path, "annotations", "vocab.txt")
    train_Xy, inference_vectorizer = preprocessor.get_train_Xy(list(preprocessor.train_document_ids()), sections_of_interest=None, vocabulary_file=vocab_f, include_sentence_span_splits=True)
        
    if not(USE_TEST):
        # create an internal validation set from the training data; use 90% for training and 10% for validation.
        split_index = int(len(train_Xy) * .9)
        val_Xy = train_Xy[split_index:]
        train_Xy = train_Xy[:split_index]
        test_Xy = preprocessor.get_Xy(preprocessor.validation_document_ids(), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
    else:
        val_Xy = preprocessor.get_Xy(preprocessor.validation_document_ids(), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
        test_Xy = preprocessor.get_Xy(preprocessor.test_document_ids(), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
    
    if USE_CUDA:
        se_scn = ScanNet(inference_vectorizer, use_attention=use_attn).cuda()
    else:
        se_scn = ScanNet(inference_vectorizer, use_attention=use_attn)
     
    
    # train with 50 epochs, batch_size of 1, and patience of 3 (early stopping)
    train_scan(se_scn, inference_vectorizer, train_Xy, val_Xy, test_Xy, 50, 32, 10)
    
    # save to specified path
    torch.save(se_scn.state_dict(), loc)

