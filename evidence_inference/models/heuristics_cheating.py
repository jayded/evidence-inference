# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:59:23 2018

@author: Eric
"""
from os.path import join, dirname, abspath
import sys
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import evidence_inference.preprocess.preprocessor as preprocessor
from evidence_inference.models.heuristics import try_except_parse, eval_sentence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


"""
Loads in the text, prompt, and answers, and gives it to the functions to find
an answer + evidence.

"""
def main():
    # Load in the data
    prompts = np.asarray(preprocessor.read_prompts())
    annotations = np.asarray(preprocessor.read_annotations())
    data = {}
    preds, y_test = [], []

    
    # store data in dictionary
    for p in prompts:
        data[p[0]] = {'xml': p[1], 'outcome': p[2], 'intervention': p[3], 'comparator': p[4], 'answer': '', 'reasoning': ''}
        
    for a in annotations:
        if (a[3]):
            data[a[1]]['answer'] = a[7]
        if (a[4]):
            data[a[1]]['reasoning'] += str(a[6])
       
    test_id = preprocessor.test_document_ids()
    # get predictions and add them to array
    for k in data.keys():
        # try to parse text to remove weird things
        id_   = data[k]['xml']
        
        # if the file is not a test file
        if not(id_ in test_id):
            continue 
        
        out   = try_except_parse(data[k]['outcome'])
        inter = try_except_parse(data[k]['intervention'])
        cmp   = try_except_parse(data[k]['comparator'])
        ans   = try_except_parse(data[k]['answer'])
        res   = try_except_parse(data[k]['reasoning'])
        
        if (ans == ''):
            continue # we don't have a valid answer for this one... 
            
        y_test.append(ans)

        # just use the reasoning as our sentence        
        likely_sentence = res
        guess = eval_sentence(likely_sentence, out, inter, cmp)
        
        if (guess == "No significant difference"):
            preds.append(0)
        elif (guess == "Significantly decreased"):
            preds.append(-1)
        else:
            preds.append(1)
         
    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds, average='macro')
    prec = precision_score(y_test, preds, average = 'macro')
    rec  = recall_score(y_test, preds, average = 'macro')
        
    return acc, f1, prec, rec

if __name__ == '__main__':
    print(main())    
