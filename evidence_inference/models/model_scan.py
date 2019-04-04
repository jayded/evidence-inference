# -*- coding: utf-8 -*-
"""
Neural sentence tagging model to be used to identify sentences that contain evidence spans.

Intended for use in a pipelined approach, where we first tag sentences and then pass those
predicted to contain evidence spans forward for inference.
"""
from random import randint
import numpy as np
from os.path import join, dirname, abspath
import sys
import argparse
from evidence_inference.preprocess.preprocessor import SimpleInferenceVectorizer as SimpleInferenceVectorizer

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from evidence_inference.models.model_ico_scan import scan_reform

import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from evidence_inference.models.model_0 import InferenceNet, GRUEncoder, _get_y_vec, PaddedSequence

USE_CUDA = True

class ScanNet(nn.Module):
    
    def __init__(self, inference_vector, use_attention=False, bi_GRU = True):
        super(ScanNet, self).__init__()
        self.vectorizer = inference_vector
        init_embedding_weights = InferenceNet.init_word_vectors("embeddings/PubMed-w2v.bin", inference_vector)
        vocab_size = len(self.vectorizer.idx_to_str)
        self.use_attention = use_attention
        self.sen_encoder = GRUEncoder(vocab_size=vocab_size, use_attention=self.use_attention, 
                                      hidden_size=32, embeddings=init_embedding_weights,
                                      bidirectional = bi_GRU)
        self.sen_encoder.embedding.requires_grad = False
        self.sen_encoder.embedding.weight.requires_grad = False

        self.out = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        
    def _encode(self, s_token):
        s_v = None
        if self.use_attention:
            _, s_v, attn_weights = self.sen_encoder(s_token)
        else:
            _, s_v = self.sen_encoder(s_token)
        return s_v
        
    def forward(self, sentence_token, batch_size = 1, h_dropout_rate = .2):
        """ 
        Negative sampling 
        -> 
        for each article, take 1 negative and 1 positive randomly, move on.
        """           
        # Encode the sentences
        s_v = self._encode(sentence_token)
        
        # Get an output
        raw_out = self.out(s_v)

        return self.sig(raw_out)
    
def early_stopping(loss, patience):
    """ Samples the last N = patience losses and returns true if all of them are in decrementing order. """
    if (len(loss) < patience):
        return False
    
    
    loss = loss[(patience * -1):]
    last = sys.float_info.min # smallest possible value
    for l in loss: 
        # if this loss is less than before
        if l < last:
            return False 
        
        last = l
        
    return True
            

def train_reformat(train_Xy):
    """
    Group train_Xy into articles, with a dictionary containing true and false rows
    """
    xy = {}
    for row in train_Xy:
        a_id = row['a_id']
        sentences = row['sentence_span']
        t = [] # for when the labels are true
        f = [] # for when the labels are false
        for s in sentences:
            if (s[1] == 0):
                f.append(s[0])
            else:
                t.append(s[0])

        if (a_id in xy):
            for sample in t:
                xy[a_id]['true'].append(sample) 
            for sample in f:
                xy[a_id]['false'].append(sample)
        else:
            # if this article is not already in the dictionary, then 
            # initialize 
            xy[a_id] = {}
            xy[a_id]['true'] = t
            xy[a_id]['false'] = f
            xy[a_id]['sample'] = 0 # location of which true value was last used.
            
    return xy

def sample_train(train_Xy):
    """ 
    For each article, on each epoch start, sample 2 to put into the training array.
    """
    data = []
    for key in train_Xy.keys():
        article = train_Xy[key]
        # for the case where there is either no true/false spans.
        if (len(article['true']) == 0 or len(article['false']) == 0):
            continue 
        
        #loc_t  = article['sample'] # location of which true value was last used.

        ## 
        # take all the positive examples (evidence spans)
        for e_span_sent in article['true']:
            if len(e_span_sent) > 0:
                data.append({'sentence_span': e_span_sent, 'y': 1})
              
                # sample a negative instance
                neg_idx  = randint(0, len(article['false'])-1)
                # make sure it's not empty.
                if len(article['false'][neg_idx]) > 0:
                    data.append({'sentence_span': article['false'][neg_idx], 'y': 0})

        '''
        #loc_f  = randint(0, len(article['false'])-1)
        #tr_val = {'sentence_span': article['true'][loc_t], 'y': 1}
        #fa_val = {'sentence_span': article['false'][loc_f], 'y': 0}
         
        # add to data
        if (len(tr_val['sentence_span']) > 0):
            data.append(tr_val)
            
        if (len(fa_val['sentence_span']) > 0):   
            data.append(fa_val)
        
        # reset the sample
        #article['sample'] = (article['sample'] + 1) % len(article['true'])
        '''
    return data

def scan_reform(data):
    """ Reformat the data to only contain what we want """
    xy = []
    for row in data:           
        sentences = row['sentence_span']
        i = -1
        for s in sentences:
            i += 1
            if (len(s[0]) != 0):
                xy.append({"sentence_span": s[0], "y" : s[1], "token_ev_labels": row['token_ev_labels'][i]})
            
    return xy


def train_scan(scan_net, inference_vectorizer, train_Xy, val_Xy, test_Xy, epochs = 1, batch_size = 1, patience = 3):    
    train_Xy, val_Xy, test_Xy = train_reformat(train_Xy), scan_reform(val_Xy), scan_reform(test_Xy) 
    
    optimizer = optim.Adam(scan_net.parameters())
    criterion = nn.BCELoss(reduction='sum')  # sum (not average) of the batch losses.
    total_epoch_loss = []
    
    for epoch in range(epochs):
        if (early_stopping(total_epoch_loss, patience)):
            break 
        
        epoch_loss = 0
        epoch_samples = sample_train(train_Xy) # sample a reasonable amount of 
        for i in range(0, len(epoch_samples), batch_size):
            instances = epoch_samples[i:i+batch_size]
            ys = torch.FloatTensor([inst['y'] for inst in instances])
                 
            unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
            sentences = [torch.LongTensor(inst["sentence_span"]) for inst in instances]
            sens, = [PaddedSequence.autopad(sentences, batch_first=True, padding_value=unk_idx)]
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
            y_vec = torch.FloatTensor([inst['y'] for inst in val_Xy])
            instances = val_Xy
            if (USE_CUDA):
                y_vec = y_vec.cuda()
                
            unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
            y_val_vec = [] # predictions
            # we batch this so the GPU doesn't run out of memory
            for i in range(0, len(instances), batch_size):
                batch_instances = instances[i:i+batch_size]
                sentences = [torch.LongTensor(inst["sentence_span"]) for inst in batch_instances]
                sens, = [PaddedSequence.autopad(sentences, batch_first=True, padding_value=unk_idx)]
                if (USE_CUDA):
                    sens = sens.cuda()
                    
                y_val_preds = scan_net(sens, batch_size=len(batch_instances))
                
                for y_v in y_val_preds:
                    y_val_vec.append(y_v)
                
            y_preds = torch.FloatTensor(y_val_vec).cuda()
            val_loss = criterion(y_preds, y_vec)
            y_hat = [1 if y > .5 else 0 for y in y_preds]
            
            y_vec = y_vec.cpu()
            acc = accuracy_score(y_vec, y_hat)
            f1  = f1_score(y_vec, y_hat)
            prc = precision_score(y_vec, y_hat)
            rc  = recall_score(y_vec, y_hat)
            

            print("epoch {}. train loss: {}; val loss: {}; val acc: {:.3f}; val f1: {:.3f}; val precision: {:.3f}; val recall: {:.3f}".format(
                        epoch, epoch_loss, val_loss, acc, f1, prc, rc))  
            total_epoch_loss.append(val_loss)
    
    return scan_net
