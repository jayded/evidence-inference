# -*- coding: utf-8 -*-
import torch
import argparse
import os
import sys
import random
import numpy as np

from os.path import join, dirname, abspath

#parser = argparse.ArgumentParser(description="Run scan-net and save to given location")
#parser.add_argument('--path', dest='path', default='./scan_model_ICO.pth')
mp0 = os.path.abspath(os.path.join('../.././evidence_inference/'))
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..'))) 
from evidence_inference.preprocess import preprocessor
USE_CUDA = True
USE_TEST = True     

use_attn = False

from evidence_inference.models.model_ico_scan import ScanNet, train_scan, scan_reform
from evidence_inference.preprocess.preprocessor import SimpleInferenceVectorizer as SimpleInferenceVectorizer
from evidence_inference.models.model_0 import PaddedSequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def test_model(scan_net, test_Xy, inference_vectorizer):
    test_Xy = scan_reform(test_Xy)
    with torch.no_grad():    
        instances = test_Xy   
        y_test = torch.FloatTensor([inst['y'] for inst in instances])
        if (USE_CUDA):
            y_test = y_test.cuda()     
            
        unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
        y_preds = [] # predictions
        # we batch this so the GPU doesn't run out of memory
        token_predictions = []
        token_label = []
        for i in range(0, len(instances)):
            batch_instances = instances[i:i+1] # batch size of 1
            sentences = [torch.LongTensor(inst["sentence_span"]) for inst in batch_instances]
            
            I = [torch.LongTensor(inst["I"]) for inst in batch_instances]
            C = [torch.LongTensor(inst["C"]) for inst in batch_instances]
            O = [torch.LongTensor(inst["O"]) for inst in batch_instances]
            t_labels = batch_instances[0]['token_ev_labels']
            sens, I, C, O = [PaddedSequence.autopad(to_enc, batch_first=True, padding_value=unk_idx) for to_enc in [sentences, I, C, O]]
            
            if USE_CUDA:
                sens = sens.cuda()
                I    = I.cuda()
                C    = C.cuda()
                O    = O.cuda()
                
            preds = scan_net(sens, I, C, O)
            
            for p in preds:
                y_preds.append(p)
    
            for j in range(len(sentences)):                
                for k in range(len(sentences[j])):
                    token_predictions.append(preds[j])
                    token_label.append(t_labels[k])
            
        y_preds = torch.FloatTensor(y_preds).cuda()
        y_bin = [1 if y > .5 else 0 for y in y_preds]
        
        auc = roc_auc_score(token_label, token_predictions) # token auc
        acc = accuracy_score(y_test, y_bin)
        f1  = f1_score(y_test, y_bin)
        prc = precision_score(y_test, y_bin)
        rc  = recall_score(y_test, y_bin)
        
        return acc, f1, prc, rc, auc

def run_scan_net_ico(loc = "scan_net_ICO_no_attn_test.pth"):
    print("Modules loaded.")
    
    parent_path = abspath(os.path.join(dirname(abspath(__file__)), '..', '..'))
    vocab_f = os.path.join(parent_path, "annotations", "vocab.txt")
    
    
    train_Xy, inference_vectorizer = preprocessor.get_train_Xy(list(preprocessor.train_document_ids()), sections_of_interest=None, 
                                                               vocabulary_file=vocab_f,
                                                               include_sentence_span_splits=True)
        
    print("Train Data Achieved")  
    if not(USE_TEST):
        # create an internal validation set from the training data; use 90% for training and 10% for validation.
        split_index = int(len(train_Xy) * .9)
        val_Xy = train_Xy[split_index:]
        train_Xy = train_Xy[:split_index]
        test_Xy = preprocessor.get_Xy(list(preprocessor.validation_document_ids()), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
    else:
        val_Xy = preprocessor.get_Xy(preprocessor.validation_document_ids(), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
        test_Xy = preprocessor.get_Xy(preprocessor.test_document_ids(), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
    
    print("Test Data Achieved")  
    
    if USE_CUDA:
        se_scn = ScanNet(inference_vectorizer, use_attention=use_attn).cuda()
    else:
        se_scn = ScanNet(inference_vectorizer, use_attention=use_attn)
        
    print("Model loaded")
    # train with 50 epochs, batch_size of 1, and patience of 3 (early stopping)
    train_scan(se_scn, inference_vectorizer, train_Xy, val_Xy, 50, 32, 10)
    
    acc, f1, prc, rc, auc = test_model(se_scn, test_Xy, inference_vectorizer)
    
    # save to specified path
    #args = parser.parse_args()
    torch.save(se_scn.state_dict(), loc)


