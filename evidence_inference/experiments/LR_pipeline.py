# -*- coding: utf-8 -*-
import random
import copy
from os.path import join, dirname, abspath
import sys

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import torch
import numpy as np
import pandas as pd
import evidence_inference.preprocess.preprocessor as preprocessor
from evidence_inference.models.regression import bag_of_words, train_model, test_model
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from evidence_inference.models.scan_regression import ScanNet
from evidence_inference.models.scan_regression import train_reformat, scan_reform, Bag_of_words

# just set all the seeds!
np.random.seed(500)
random.seed(500)
torch.manual_seed(500)
    
    
PROMPT_ID_COL_NAME = "PromptID"
LBL_COL_NAME = "Label Code"
EVIDENCE_COL_NAME = "Annotations"
STUDY_ID_COL = "PMCID"

USE_CUDA = False

def load_model_scan(inference_vectorizer, loc = './model_lr.pth'):
    """ Load in the model (with proper weights). """
    # Note: here we are manually setting use_attention = False,
    # but this should really be a parameter!
    sn = ScanNet(inference_vectorizer)
    # We partially load parameters because attention weights
    # will not exist if attention was shut off.
    state = sn.state_dict()
    partial = torch.load(loc)
    state.update(partial)
    sn.load_state_dict(state)
    if (USE_CUDA):
        sn = sn.cuda()
    sn.eval()
    return sn

def get_preds(model, span, inference_vectorizer):
    """ Get a prediction from the model for a single span. """
    if len(span) == 0:
        # if we happen to get an empty span, predict 0.
        return 0 
    batch_instances = [span]
    sens = torch.FloatTensor(batch_instances)
    if USE_CUDA:
        sens = sens.cuda()
    preds = model(sens, batch_size=1)
    pred = preds[0].data.tolist()[0]
    return pred

def reformat(Xy, inference_vectorizer, model):
    bow = Bag_of_words(inference_vectorizer)
    x = []
    y = []
    for prompt in Xy:
        art = prompt['sentence_span']
        out = bow.transform(prompt['O'])
        itv = bow.transform(prompt['I'])
        cmp = bow.transform(prompt['C'])

        y.append([prompt['y'][0][0]])
        new_sen = [0] * len(inference_vectorizer.idx_to_str)
        
        for s in art:
            s = s[0]
            bow_rep = bow.transform(s)
            pred = get_preds(model, bow_rep, inference_vectorizer)
            if (pred > 0.5):
                # add this sentence to the document 
                for i in range(len(bow_rep)):
                    if (bow_rep[i] == 1):
                        new_sen[i] = 1
        
        # after we found our sentences
        x.append(np.append(np.append(np.append(new_sen, bow.transform(out)), bow.transform(itv)), bow.transform(cmp)))
                
    x, y = torch.FloatTensor(x), torch.LongTensor(y)
    if USE_CUDA:
        x = x.cuda()
        y = y.cuda()
                
    return x, y

def load_data(use_test, model_loc):
    """
    Load the data into a train/val/test set that allows for easy access.

    @return bag-of-word representation of training, validation, test sets (with labels).
    """    

    t_ids = set(list(preprocessor.train_document_ids()))
    te_ids = set(list(preprocessor.test_document_ids()))
    val_ids = set(list(preprocessor.validation_document_ids()))
    train_Xy, inference_vectorizer = preprocessor.get_train_Xy(t_ids, sections_of_interest=None, vocabulary_file=None, include_sentence_span_splits = True)

    # load model
    model = load_model_scan(inference_vectorizer, model_loc)
        
    # create an internal validation set from the training data; use 90% for training and 10% for validation.
    random.shuffle(train_Xy)
    
    if not(use_test):
        split_index = int(len(train_Xy) * .9)
        val_Xy = train_Xy[split_index:]
        train_Xy = train_Xy[:split_index]
        test_Xy = preprocessor.get_Xy(set(list(preprocessor.validation_document_ids())), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
    else:
        val_Xy = preprocessor.get_Xy(val_ids, inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
        test_Xy = preprocessor.get_Xy(te_ids, inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
        
    
    x_train, y_train = reformat(train_Xy, inference_vectorizer, model)
    x_val, y_val     = reformat(val_Xy, inference_vectorizer, model)
    x_test, y_test   = reformat(test_Xy, inference_vectorizer, model)
    return x_train, y_train, x_val, y_val, x_test, y_test
    

def run_lr_pipeline(iterations, use_test, path = './model_lr.pth'):
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(use_test, path)
    y_test += 1
    print("Loaded {} training examples, {} validation examples, {} testing examples".format(len(x_train), len(x_val), len(x_test)))
    model = train_model(x_train, y_train, x_val, y_val, iterations, learning_rate=0.001)
    preds = test_model(model, x_test)
    
    preds = preds
    y_test = y_test.cpu()
    # calculate f1 and accuracy
    print(classification_report(y_test, preds))
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    prec = precision_score(y_test, preds, average = 'macro')
    rec  = recall_score(y_test, preds, average = 'macro')
    print(acc)
    print(f1)
    print(prec)
    print(rec)
    print("\n\n")

    return acc, f1, prec, rec

if __name__ == "__main__":
    print(run_lr_pipeline(1, True))
