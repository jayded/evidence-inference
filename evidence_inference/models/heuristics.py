# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:59:23 2018

@author: Eric
"""
from os.path import join, dirname, abspath
import sys
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import re
import string
import numpy as np
from functools import reduce
from pattern.en import lexeme
from nltk.corpus import wordnet
from evidence_inference.experiments.model_0_paper_experiment import get_data
import evidence_inference.preprocess.preprocessor as preprocessor
from evidence_inference.models.heuristic_utils import parse_p_value, sig_or_ns_pval, split_sentences, strip_punctuation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_token_mass(token_labels, token_preds):
    all_token_mass = []
    for label, art_pred in zip(token_labels, token_preds):
        pred = []
        for sen_l, sen_p in zip(label, art_pred):
            pred.append([sen_p] * len(sen_l))
            
    
            
        total = np.sum(np.sum(pred)); pred = [np.asarray(p) / total for p in pred] # normalize
        token_mass = 0
        for i in range(len(label)):
            for j in range(len(label[i])): 
                if label[i][j] == 1:
                    try:
                        token_mass += pred[i][j]
                    except: 
                        import pdb; pdb.set_trace()
                else:
                    pass
        
        all_token_mass.append(token_mass)
    
    return np.mean(all_token_mass)

"""
Creates a dictionary of word occurances.

@param str sentence is a sentence
@returns a dictionary relating to the frequency of words in the given sentence.
"""
def generate_word_dict(sentence):
    sentence = strip_punctuation(sentence)    
    translator = str.maketrans('','',string.punctuation)
    sentence = sentence.translate(translator)
    return set(sentence.split(" "))

"""
Locates and returns the sentence that likely contains the evidence.

@param str text represents a string representation of the text.
@param str out represents the outcome.
@param str inter represents the intervention.
@param str cmp represents the comparator.
@returns a sentence that likely contains the answer.
"""
def locate_probable_sentence(text, out, inter, cmp):
    sentences = split_sentences(text) 
    sentences = list(filter((lambda x: x != ''), sentences))
    dict_of_words = list(map(generate_word_dict, sentences))
    
    point_arr = [None] * len(sentences)
    idx = 0
    for d in dict_of_words:
        points = 0
        points += reduce((lambda y, x: y + 1 if x in d else y), out.split(" "), 0)
        points += reduce((lambda y, x: y + 1 if x in d else y), inter.split(" "), 0)
        points += reduce((lambda y, x: y + 1 if x in d else y), inter.split(" "), 0)
        point_arr[idx] = points
        idx += 1
        
    loc_best = np.argmax(point_arr)
    likely_sentence = sentences[loc_best]

    return likely_sentence, point_arr
                 
"""
Tries to infer sig inc./dec. based off of word choice.

@param str sen is the sentence that needs to have direction inferred upon.
@param str default is what it returns if it can't find anything useful.
@return sig inc/dec. or default.
"""
def infer_direction(sen, default):
    all_nw = []
    all_pw = []
    
    nw = ["decrease"]
    pw = ["increase"]
    
    # add in all words that we want
    for i in range(len(nw)):
        neg_words = list(reduce((lambda y, x: np.append(y, x.lemma_names())), wordnet.synsets(nw[i]), []))    
        pos_words = list(reduce((lambda y, x: np.append(y, x.lemma_names())), wordnet.synsets(pw[i]), []))
        all_nw.extend(neg_words)
        all_pw.extend(pos_words)
       
    # add in different forms of the word based on english rules
    try:
        all_nw = list(reduce((lambda y, x: np.append(y, lexeme(x))), all_nw, []))
        all_pw = list(reduce((lambda y, x: np.append(y, lexeme(x))), all_pw, []))
    except:
        print("Error. Continue.")
        return infer_direction(sen, default)

        
    # remove duplicates
    all_nw = [x for x in iter(set(all_nw))]
    all_pw = [x for x in iter(set(all_pw))]
    
    neg = 0
    pos = 0
    
    # number of positive words
    for word in all_nw:
        if word in sen:
            neg += 1
    
    # number of negative words
    for word in all_pw:
        if word in sen:
            pos += 1
      
    if (pos > neg):
        return "Significantly increased"
    elif (neg > pos):
        return "Significantly decreased"
    else:
        return default
    
    
"""
Finds the p-value in the strings. If it is greater than 0.05, than return
not significant. 

@param str sen is the sentence search within.
@return whether or not something is or isn't significant.
"""
def find_p_value(sen, out, inter, cmp):
    to_find = ["p=", "p>", "p<", "p =", "p >", "p <"]
    equals = ["p=", "p ="]
    ns_vals = ["p>", "p >"]
    ns = 0
    sig = 0  
    
    # store the location in a dictionary
    for find in to_find:
        try:
            loc = [m.start() for m in re.finditer(find, sen)]
        except:
            loc = []
                
        try:
            loc_cmp = [m.start() for m in re.finditer(cmp, sen)]
        except:
            loc_cmp = []
        
        
        try:
            loc_inter = [m.start() for m in re.finditer(inter, sen)]
        except:
            loc_inter = []

        # find the closest p-value if the comparator is mentioned.
        if (find in equals and len(loc_cmp) > 0):
            ns, sig = sig_or_ns_pval(sen, find, loc_cmp[0], loc, ns, sig)
            
        # find the closest p-value if the intervention is mentioned
        elif (find in equals and len(loc_inter) > 0):
            ns, sig = sig_or_ns_pval(sen, find, loc_inter[0], loc, ns, sig)
            
        # if neither are mentioned
        elif (find in equals):
            all_val = list(map((lambda l: parse_p_value(sen[l + len(find):])), loc)) # get p-values
            all_val = list(filter((lambda x: x != -1), all_val)) # remove any errors
            count_ns = len(list(filter((lambda x: x > 0.05), all_val)))
            count_sig = len(loc) - count_ns
            ns += count_ns
            sig += count_sig
            
        # We know that it is bigger than 0.05
        elif (find in ns_vals):
            ns += len(loc)
            
        # If (p < X) is mentioned, then probably there is significance involved
        else:
            sig += len(loc)
    
    # there are more significant p-values        
    if (sig > ns):
        return infer_direction(sen, "Significantly increased")
        
    # No p-values found
    elif (sig == 0 and ns == 0):
        return infer_direction(sen, "No significant difference")
    
    # Non-significance
    else:
        return "No significant difference"
    
        
"""
Evaluates the sentence, and attempts to return an answer of increase/decrease/no difference.

@param str out represents the outcome.
@param str inter represents the intervention.
@param str cmp represents the comparator.
@returns an answer about how the intervention relates to the outcome in comparison to the comparator.
"""
def eval_sentence(s, out, inter, cmp):
    return find_p_value(s, out, inter, cmp)

"""
Try and except functions for 1 line code in order to parse data
"""
def try_except_parse(text):
    try:
        return text.encode('cp1252').decode('utf-8')
    except:
        return text

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
    sentences = []
    for p in prompts:
        data[p[0]] = {'xml': p[1], 'outcome': p[2], 'intervention': p[3], 'comparator': p[4], 'answer': '', 'reasoning': ''}
        
    for a in annotations:
        if (a[3]):
            data[a[1]]['answer'] = a[7]
        if (a[4]):
            data[a[1]]['reasoning'] += str(a[6]) + "; "
       
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
        
        if (ans == ''):
            continue # we don't have a valid answer for this one... 
            
        y_test.append(ans)

        
        # article text
        article = preprocessor.get_article(id_)
        text = preprocessor.extract_raw_text(article).lower()
        
        likely_sentence, pt_array = locate_probable_sentence(text, out, inter, cmp)
        guess = eval_sentence(likely_sentence, out, inter, cmp)
        
        sentences.append(pt_array)
        
        if (guess == "No significant difference"):
            preds.append(0)
        elif (guess == "Significantly decreased"):
            preds.append(-1)
        else:
            preds.append(1)
   
    # tm = calculate_token_mass(t_labels, sentences)
    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds, average='macro')
    prec = precision_score(y_test, preds, average = 'macro')
    rec  = recall_score(y_test, preds, average = 'macro')
        
    return acc, f1, prec, rec

if __name__ == '__main__':
    print(main())    
