# -*- coding: utf-8 -*-
from random import randint
import numpy as np
from os.path import join, dirname, abspath
import sys

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import argparse
from evidence_inference.preprocess.preprocessor import SimpleInferenceVectorizer as SimpleInferenceVectorizer

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from evidence_inference.models.model_0 import InferenceNet, GRUEncoder, _get_y_vec, PaddedSequence

from evidence_inference.models.model_scan import sample_train, train_reformat, scan_reform, early_stopping

USE_CUDA = True

class ScanNet(nn.Module):
    
    def __init__(self, vectorizer):
        super(ScanNet, self).__init__()   
        self.vectorizer = vectorizer
        vocab_size = len(self.vectorizer.idx_to_str)
        self.out = nn.Linear(vocab_size, 1)
        self.sig = nn.Sigmoid()
        
        
    def forward(self, sentence_token, batch_size = 1, h_dropout_rate = .2):
        """ 
        Negative sampling 
        -> 
        for each article, take 1 negative and 1 positive randomly, move on.
        """           
        return self.sig(self.out(sentence_token))

class Bag_of_words():
    
    def __init__(self, inference_vectorizer):
        self.inference_vectorizer = inference_vectorizer
        vocab_size = len(inference_vectorizer.idx_to_str)
        self.vocab_size = vocab_size
        
    def transform(self, sentence):
        ret = [0] * self.vocab_size
        for val in sentence:
            try:
                ret[val] = 1
            except:
                import pdb; pdb.set_trace()
        return ret


def train_scan(inference_vectorizer, train_Xy, val_Xy, test_Xy, epochs = 1, batch_size = 1, patience = 3):    
    scan_net = ScanNet(inference_vectorizer) #(np.asarray(val_Xy[0]['sentence_span']).shape[0], 2)
    if USE_CUDA:
        scan_net = scan_net.cuda() 
 
    optimizer = torch.optim.SGD(scan_net.parameters(), lr = 0.01)
    criterion = nn.BCELoss(reduction = 'sum') #criterion = nn.CrossEntropyLoss(reduction='sum')  # sum (not average) of the batch losses.
    bow = Bag_of_words(inference_vectorizer)
    total_epoch_loss = []
    
    for epoch in range(epochs):
        if (early_stopping(total_epoch_loss, patience)):
            break 
        
        epoch_loss = 0
        # for each article, sample data w/ a label of 0, and label of 1.
        epoch_samples = sample_train(train_Xy) 
        for i in range(0, len(epoch_samples), batch_size):
            instances = epoch_samples[i:i+batch_size]
            ys = torch.FloatTensor([inst['y'] for inst in instances])
            sens = torch.FloatTensor([bow.transform(inst["sentence_span"]) for inst in instances])
            optimizer.zero_grad()
            
            if USE_CUDA:
                sens = sens.cuda()
                ys = ys.cuda()
             
            tags = scan_net(sens)
            loss = criterion(tags, ys)
            
            # if loss.item is nan
            if loss.item() != loss.item():
                import pdb; pdb.set_trace()
                
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        with torch.no_grad():            
            instances = val_Xy
            y_hat = [] # predictions
            val_loss = 0
            y_true = [inst['y'] for inst in val_Xy]
            # we batch this so the GPU doesn't run out of memory
            for i in range(0, len(instances), batch_size):
                batch_instances = instances[i:i+batch_size]
                sens = torch.FloatTensor([bow.transform(inst['sentence_span']) for inst in batch_instances])
                ys = torch.FloatTensor([inst['y'] for inst in batch_instances])

                if (USE_CUDA):
                    sens = sens.cuda()
                    ys   = ys.cuda()
                    
                tags = scan_net(sens, batch_size=len(batch_instances))
                val_loss += criterion(tags, ys)
                #_, preds = torch.max(tags.data, 1)
                y_hat = np.append(y_hat, tags.data.cpu().numpy())
                
                    
            y_hat = [1 if y > .5 else 0 for y in y_hat]
            acc = accuracy_score(y_true, y_hat)
            f1  = f1_score(y_true, y_hat)
            prc = precision_score(y_true, y_hat)
            rc  = recall_score(y_true, y_hat)
            
            print("epoch {}. train loss: {:.2f}; val loss: {:.2f}; val acc: {:.2f}; val f1: {:.2f}; val precision: {:.2f}; val recall: {:.2f}".format(
                        epoch, epoch_loss, val_loss, acc, f1, prc, rc))  
            total_epoch_loss.append(val_loss)
    
    return scan_net
