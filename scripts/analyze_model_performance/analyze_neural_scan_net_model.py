# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:43:24 2018

@author: Eric
"""
import os
import sys

# load path
mp1 = os.path.abspath(os.path.join('..//..//.//'))

module_paths = [mp1]
for mp in module_paths:
    if mp not in sys.path:
        sys.path.append(mp)
        
import torch
import random
from evidence_inference.preprocess import preprocessor
from evidence_inference.models.model_scan import ScanNet, scan_reform
from evidence_inference.models.model_0 import PaddedSequence
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from evidence_inference.preprocess.preprocessor import SimpleInferenceVectorizer as SimpleInferenceVectorizer

USE_CUDA = True
USE_TEST = True

def load_model_scan(inference_vectorizer, loc):
    ''' Load in the model (with proper weights). '''
    # Note: here we are manually setting use_attention = False,
    # but this should really be a parameter!
    sn = ScanNet(inference_vectorizer, use_attention=True)
    # We partially load parameters because attention weights
    # will not exist if attention was shut off.
    state = sn.state_dict()
    partial = torch.load(loc)
    state.update(partial)
    sn.load_state_dict(state)
    sn = sn.cuda()
    sn.eval()
    return sn

def get_preds(model, span, inference_vectorizer):
    ''' Get a prediction from the model for a single span. '''
    if len(span) == 0:
        # if we happen to get an empty span, predict 0.
        return 0 
    batch_instances = [span]
    unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
    sentences = [torch.LongTensor(inst) for inst in batch_instances]
    sens, = [PaddedSequence.autopad(sentences, batch_first=True, padding_value=unk_idx)]
    sens = sens.cuda()
    preds = model(sens, batch_size=len(sentences))
    pred = preds[0].data.tolist()[0]
    return pred

print("Loading data...")
# get training data
train_Xy, inference_vectorizer = preprocessor.get_train_Xy(set(list(preprocessor.train_document_ids())), sections_of_interest=None, vocabulary_file="..//..//.//annotations/vocab.txt", include_sentence_span_splits = True)
print("Training data loaded...")  

if not(USE_TEST):
    split_index = int(len(train_Xy) * .9)
    val_Xy = train_Xy[split_index:]
    train_Xy = train_Xy[:split_index]
    test_Xy = preprocessor.get_Xy(set(list(preprocessor.validation_document_ids())), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
else:
    val_Xy = preprocessor.get_Xy(set(list(preprocessor.validation_document_ids())), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
    test_Xy = preprocessor.get_Xy(set(list(preprocessor.test_document_ids())), inference_vectorizer, sections_of_interest=None, include_sentence_span_splits = True) 
        
print("Test data loaded...")  

# modify training data
val_Xy, test_Xy = scan_reform(val_Xy), scan_reform(test_Xy) 
print("Reformatted data.")

# load the model
model = load_model_scan(inference_vectorizer, './models/scan_model_neural.pth')
print("Model loaded...")  

# after loading the model, get all predictions.
instances = val_Xy # validation set for now... 
y_preds = []
y_test  = []
token_predictions = []
token_label = []

# validation statistics
with torch.no_grad():
    for i in range(0, len(instances)):
        y_vec = torch.FloatTensor([inst['y'] for inst in val_Xy])
        if (USE_CUDA):
            y_vec = y_vec.cuda()
            
        # we batch this so the GPU doesn't run out of memory
        for i in range(0, len(instances)):
            span = instances[i]["sentence_span"]
            y    = instances[i]["y"]
            t_labels = instances[i]['token_ev_labels']

            pred = get_preds(model, span, inference_vectorizer)
            y_preds.append(pred)
            for i in range(len(span)):
                token_predictions.append(pred)
                token_label.append(t_labels[i])
                
# get the statistics
preds = [1 if y > .5 else 0 for y in y_preds]
t_ac = roc_auc_score(token_label, token_predictions) # token auc
auc  = roc_auc_score(y_test, y_preds)
acc  = accuracy_score(y_test, preds)
f1   = f1_score(y_test, preds, average='macro')
prec = precision_score(y_test, preds, average = 'macro')
rec  = recall_score(y_test, preds, average = 'macro')
  
# print results
print("Overall AUC: {:.2f}".format(auc))
print("Token AUC: {:.3f}".format(t_ac))
print("F1: {:.2f}".format(f1))
print("Precision: {:.2f}".format(prec))
print("Recall: {:.2f}".format(rec))
