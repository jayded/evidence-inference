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


import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from evidence_inference.models.model_0 import InferenceNet, CBoWEncoder, GRUEncoder, _get_y_vec, PaddedSequence

USE_CUDA = True

class ScanNet(nn.Module):
    
    def __init__(self, inference_vector, use_attention=False, 
                 condition_attention = False, bi_GRU = True, 
                 ICO_encoder = "CBOW", h_size_ICO = 32, h_size = 32):
        
        super(ScanNet, self).__init__()
        self.vectorizer = inference_vector
        init_embedding_weights = InferenceNet.init_word_vectors("./embeddings/PubMed-w2v.bin", inference_vector)
        vocab_size = len(self.vectorizer.idx_to_str)
        
        print("Loading Article encoder...")
        self.ICO_encoder = ICO_encoder
        self.condition_attention = condition_attention
        self.use_attention = use_attention
        
        
        print("Loading ICO encoders...")
        if ICO_encoder == "CBOW":
            self.intervention_encoder = CBoWEncoder(vocab_size=vocab_size, embeddings=init_embedding_weights)
            self.comparator_encoder = CBoWEncoder(vocab_size=vocab_size, embeddings=init_embedding_weights)
            self.outcome_encoder = CBoWEncoder(vocab_size=vocab_size, embeddings=init_embedding_weights)
            self.ICO_dims = init_embedding_weights.embedding_dim * 3
            self.MLP_input_size = self.ICO_dims + h_size
        elif ICO_encoder == "GRU" or ICO_encoder == "BIGRU":
            self.intervention_encoder = GRUEncoder(vocab_size=vocab_size, 
                                                   hidden_size=h_size_ICO,
                                                   embeddings=init_embedding_weights,
                                                   bidirectional = ICO_encoder == "BIGRU")
            self.comparator_encoder = GRUEncoder(vocab_size=vocab_size, 
                                                 hidden_size=h_size_ICO,
                                                 embeddings=init_embedding_weights,
                                                 bidirectional = ICO_encoder == "BIGRU")
            self.outcome_encoder = GRUEncoder(vocab_size=vocab_size, 
                                              hidden_size=h_size_ICO,
                                              embeddings=init_embedding_weights,
                                              bidirectional = ICO_encoder == "BIGRU")
            self.MLP_input_size = 0
        
        self.sen_encoder = GRUEncoder(vocab_size=vocab_size, use_attention=self.use_attention, 
                                      hidden_size=h_size, embeddings=init_embedding_weights,
                                      bidirectional = bi_GRU, condition_attention = condition_attention, query_dims=self.ICO_dims)   
        
        # freeze embeddings
        for layer in (
                    self.sen_encoder, self.intervention_encoder, self.comparator_encoder, self.outcome_encoder):
                # note: we are relying on the fact that all encoders will have a
                # "embedding" layer (nn.Embedding). 
                layer.embedding.requires_grad = False
                layer.embedding.weight.requires_grad = False

        self.out = nn.Linear(self.MLP_input_size, 1)
        self.sig = nn.Sigmoid()
        
    def _encode(self, I_tokens, C_tokens, O_tokens):
        if self.ICO_encoder == "CBOW":
            # simpler case of a CBoW encoder.
            I_v = self.intervention_encoder(I_tokens)
            C_v = self.comparator_encoder(C_tokens)
            O_v = self.outcome_encoder(O_tokens)
        elif self.ICO_encoder == 'GRU' or self.ICO_encoder == 'biGRU':
            # then we have an RNN encoder. Hidden layers are automatically initialized
            _, I_v = self.intervention_encoder(I_tokens)
            _, C_v = self.comparator_encoder(C_tokens)
            _, O_v = self.outcome_encoder(O_tokens)
        else:
            raise ValueError("No such encoder: {}".format(self.ICO_encoder))
        return I_v, C_v, O_v
        
    def forward(self, sentence_token, I_tokens, C_tokens, O_tokens):   
        I_v, C_v, O_v = self._encode(I_tokens, C_tokens, O_tokens)
        if self.sen_encoder.use_attention:

            query_v = None
            if self.sen_encoder.condition_attention:
                query_v = torch.cat([I_v, C_v, O_v], dim=1)

            _, s_v, attn_weights = self.sen_encoder(sentence_token, query_v_for_attention=query_v)
        
        else:
            _, s_v = self.sen_encoder(sentence_token)
           
        
        # Get an output
        h = torch.cat([s_v, I_v, C_v, O_v], dim = 1)
        raw_out = self.out(h)

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
        p_id = row['p_id']
        sentences = row['sentence_span']
        t = [] # for when the labels are true
        f = [] # for when the labels are false
        I = row["I"]
        C = row["C"]
        O = row["O"]
        
        f = [s[0] for s in list(filter(lambda s: s[1] == 0, sentences))]
        t = [s[0] for s in list(filter(lambda s: s[1] != 0, sentences))]
        
        # if this prompt id is not already in the dictionary, then 
        # initialize 
        xy[p_id] = {"I": I, "C": C, "O": O}
        xy[p_id]['true'] = t
        xy[p_id]['false'] = f
            
    return xy

def sample_train(train_Xy):
    """ 
    For each article, on each epoch start, sample 2 to put into the training array.
    """
    data = []
    for key in train_Xy.keys():
        prompt = train_Xy[key]
        I = prompt["I"]
        C = prompt["C"]
        O = prompt["O"]
        # for the case where there is either no true/false spans.
        if (len(prompt['true']) == 0 or len(prompt['false']) == 0):
            continue 
        
        # take all the positive examples (evidence spans)
        for e_span_sent in prompt['true']:
            if len(e_span_sent) > 0:
                data.append({'sentence_span': e_span_sent, 'y': 1, "I": I, 
                             "C": C, "O": O})
              
                # sample a negative instance
                neg_idx  = randint(0, len(prompt['false'])-1)
                # make sure it's not empty.
                if len(prompt['false'][neg_idx]) > 0:
                    data.append({'sentence_span': prompt['false'][neg_idx], 'y': 0, 
                                 "I": I, "C": C, "O": O})

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
                xy.append({"sentence_span": s[0], "y" : s[1], "I": row["I"], 
                           "C": row["C"], "O": row["O"], "token_ev_labels": row['token_ev_labels'][i]})
            
    return xy


def train_scan(scan_net, inference_vectorizer, train_Xy, val_Xy, epochs = 1, batch_size = 1, patience = 3):    
    train_Xy, val_Xy = train_reformat(train_Xy), scan_reform(val_Xy)
    
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
            I = [torch.LongTensor(inst["I"]) for inst in instances]
            C = [torch.LongTensor(inst["C"]) for inst in instances]
            O = [torch.LongTensor(inst["O"]) for inst in instances]
            sens, I, C, O = [PaddedSequence.autopad(to_enc, batch_first=True, padding_value=unk_idx) for to_enc in [sentences, I, C, O]]
            
            optimizer.zero_grad()
            
            if USE_CUDA:
                sens = sens.cuda()
                I    = I.cuda()
                C    = C.cuda()
                O    = O.cuda()
                ys   = ys.cuda()
                   
            tags = scan_net(sens, I, C, O)
            loss = criterion(tags, ys) #/ torch.sum(sens.batch_lengths)
            
            # if loss.item is nan
            if loss.item() != loss.item():
                import pdb; pdb.set_trace()
                
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        with torch.no_grad():                    
            y_true = torch.FloatTensor([inst['y'] for inst in val_Xy])
            instances = val_Xy
            if (USE_CUDA):
                y_true = y_true.cuda()
                
            unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
            y_preds = [] # predictions
            # we batch this so the GPU doesn't run out of memory
            for i in range(0, len(instances), batch_size):
                batch_instances = instances[i:i+batch_size]
                sentences = [torch.LongTensor(inst["sentence_span"]) for inst in batch_instances]
                I = [torch.LongTensor(inst["I"]) for inst in batch_instances]
                C = [torch.LongTensor(inst["C"]) for inst in batch_instances]
                O = [torch.LongTensor(inst["O"]) for inst in batch_instances]
                sens, I, C, O = [PaddedSequence.autopad(to_enc, batch_first=True, padding_value=unk_idx) for to_enc in [sentences, I, C, O]]
                
                if USE_CUDA:
                    sens = sens.cuda()
                    I    = I.cuda()
                    C    = C.cuda()
                    O    = O.cuda()
                    
                y_val_preds = scan_net(sens, I, C, O)
                
                for p in y_val_preds:
                    y_preds.append(p)
                
            y_preds = torch.FloatTensor(y_preds).cuda()
            val_loss = criterion(y_preds, y_true)
            y_bin = [1 if y > .5 else 0 for y in y_preds]
        
            acc = accuracy_score(y_true, y_bin)
            f1  = f1_score(y_true, y_bin)
            prc = precision_score(y_true, y_bin)
            rc  = recall_score(y_true, y_bin)
            

            print("epoch {}. train loss: {}; val loss: {}; val acc: {:.3f}; val f1: {:.3f}; val precision: {:.3f}; val recall: {:.3f}".format(
                        epoch, epoch_loss, val_loss, acc, f1, prc, rc))  
            total_epoch_loss.append(val_loss)
    
    return scan_net
