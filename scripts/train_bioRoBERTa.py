'''
Train a RoBERTa-based (unconditional!) punchline extractor. 
This is a simple helper 
'''

import random 
import numpy as np 

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, PretrainedConfig

import sys
sys.path.append("../")
from evidence_inference.preprocess.preprocessor import get_Xy, train_document_ids, test_document_ids, validation_document_ids, get_train_Xy                                                                    

tr_ids = list(train_document_ids()) 
train_Xy, inference_vectorizer = get_train_Xy(tr_ids, sections_of_interest=None, 
                                 vocabulary_file=None, include_sentence_span_splits=False, 
                                 include_raw_texts=True)


val_ids = list(validation_document_ids())
val_Xy  = get_Xy(val_ids[:10], inference_vectorizer, include_raw_texts=True) 


def instances_from_article(article_dict, neg_samples=2, max_instances=6):

    def filter_empty(snippets):
        return [s for s in snippets if len(s)>1]

    evidence_snippets = filter_empty([snippet[1] for snippet in article_dict['y']])
    positive_snippets = evidence_snippets

    max_pos = max_instances / (neg_samples + 1)

    if len(evidence_snippets) > max_pos:
        positive_snippets = random.sample(evidence_snippets, int(max_pos))

    n_pos = len(positive_snippets)
    n_neg = n_pos * neg_samples

    all_snippets = filter_empty([snippet[-1] for snippet in article_dict['all_article_sentences']])
                      
    negative_snippets = random.sample(all_snippets, n_neg)

    instances, labels = positive_snippets + negative_snippets, [1]*n_pos + [0]*n_neg
    
    return (instances, labels)


def train(train_Xy, n_epochs=4, batch_size=4): # val_Xy
    tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base") 
    model     = RobertaForSequenceClassification.from_pretrained("allenai/biomed_roberta_base") 

    from transformers import AdamW
    optimizer = AdamW(model.parameters())

    
    best_val = np.inf
    for epoch in range(n_epochs):    
        model.train()
        print("on epoch ", epoch)
        # assemble a batch
        for i, article in enumerate(train_Xy):
            if (i % 10) == 0: 
                print ("on article", i)

            # assemble a batch
            batch_X, batch_y = [], []
            cur_batch_size = 0
            while cur_batch_size < batch_size:
                cur_X, cur_y = instances_from_article(article, max_instances=batch_size-cur_batch_size)
                batch_X.extend(cur_X)
                batch_y.extend(cur_y)

                cur_batch_size += len(cur_X)

            # 
            model.zero_grad()  

            batch_X_tensor = tokenizer.batch_encode_plus(batch_X, add_special_tokens=True, pad_to_max_length=True) #tokenizer.encode_plus(batch_X[:batch_size], add_special_tokens=True)
            batch_y_tensor = torch.tensor(batch_y)
            
     
            loss, logits  = model(torch.tensor(batch_X_tensor['input_ids']), 
                                  attention_mask=torch.tensor(batch_X_tensor['attention_mask']), 
                                  labels=torch.tensor(batch_y))

            print(loss)
            loss.backward()
            optimizer.step()

        ####
        # eval on val set
        ###
        print("evaluating on val...")
        model.eval()
        val_loss = 0
        for j, article in enumerate(val_Xy):
            val_X, val_y = instances_from_article(article, max_instances=batch_size)
            val_X_tensor = tokenizer.batch_encode_plus(val_X, add_special_tokens=True, pad_to_max_length=True) #tokenizer.encode_plus(batch_X[:batch_size], add_special_tokens=True)
            val_y_tensor = torch.tensor(val_y)
            
            loss, logits  = model(torch.tensor(val_X_tensor['input_ids']), 
                                  attention_mask=torch.tensor(val_X_tensor['attention_mask']), 
                                  labels=torch.tensor(val_y_tensor))
            
            val_loss += loss.detach().numpy()
        
        print("val loss after epoch {} is: {}".format(epoch, val_loss))
        if val_loss < best_val:
            print("new best loss: {}".format(val_loss))
            best_val = val_loss
            torch.save(model.state_dict(), "inference.model")


if __name__ == '__main__':
    train(train_Xy, batch_size=6)
                


