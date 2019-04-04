# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:43:24 2018

@author: Eric
"""
# load path
import os
import sys

mp1 = os.path.abspath(os.path.join('..//..//.//'))

module_paths = [mp1]
for mp in module_paths:
    if mp not in sys.path:
        sys.path.append(mp)

import torch

import random
from evidence_inference.preprocess import preprocessor
from evidence_inference.experiments.LR_pipeline import load_model_scan
from evidence_inference.models.scan_regression import train_reformat, scan_reform, Bag_of_words
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

USE_CUDA = True
USE_TEST = True

print("Loading data.")
# get training data
train_Xy, inference_vectorizer = preprocessor.get_train_Xy(set(list(preprocessor.train_document_ids())), sections_of_interest=None, vocabulary_file=None, include_sentence_span_splits = True)
print("Training data loaded.")  

if not(USE_TEST):
    split_index = int(len(train_Xy) * .9)
    val_Xy = train_Xy[split_index:]
    train_Xy = train_Xy[:split_index]
    test_Xy = preprocessor.get_Xy(set(list(preprocessor.validation_document_ids())), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
else:
    val_Xy = preprocessor.get_Xy(set(list(preprocessor.validation_document_ids())), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
    test_Xy = preprocessor.get_Xy(set(list(preprocessor.test_document_ids())), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
        
print("Test data loaded.")  

# modify training data
train_Xy, val_Xy, test_Xy = train_reformat(train_Xy), scan_reform(val_Xy), scan_reform(test_Xy) 
bow = Bag_of_words(inference_vectorizer)

print("Data transformed.")  

# load the model
model = load_model_scan(inference_vectorizer, './models/')
print("Model loaded.")

# after loading the model, get all predictions.
instances = val_Xy # validation set for now... 
y_preds = []
y_test  = []
token_predictions = []
token_label = []
cheating_labels = []


# validation statistics
with torch.no_grad():
    for i in range(0, len(instances)):
        batch_instances = instances[i: i + 1]
        span = batch_instances[0]['sentence_span']
        sens = torch.FloatTensor([bow.transform(inst['sentence_span']) for inst in batch_instances])
        y = batch_instances[0]['y']
        t_labels = batch_instances[0]['token_ev_labels']
            
    
        if USE_CUDA:
            sens = sens.cuda()
            
        preds = model(sens, batch_size=len(batch_instances))
        pred = preds[0].data.tolist()[0]

        y_preds.append(pred)
        y_test.append(y)
        
        for i in range(len(span)):
            token_predictions.append(pred)
            token_label.append(t_labels[i])
            cheating_labels.append(y)

# get the statistics
bin_preds = [1 if y > .5 else 0 for y in y_preds]
t_ac = roc_auc_score(token_label, token_predictions) # token auc

auc  = roc_auc_score(y_test, y_preds)
acc  = accuracy_score(y_test, preds)
f1   = f1_score(y_test, preds, average='macro')
prec = precision_score(y_test, preds, average = 'macro')
rec  = recall_score(y_test, preds, average = 'macro')

# print results
print("Overall AUC: {:.3f}".format(auc))
print("Token AUC: {:.3f}".format(t_ac))
print("F1: {:.3f}".format(f1))
print("Precision: {:.3f}".format(prec))
print("Recall: {:.3f}".format(rec))
  
cheating_token_auc = roc_auc_score(token_label, cheating_labels) # token auc
print("Cheating token auc: {:.3f}".format(cheating_token_auc))