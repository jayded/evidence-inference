# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:20:27 2018

@author: Eric
"""
from __future__ import unicode_literals, print_function
import spacy

nlp = spacy.load('en')

def gen_first_n_second_n(first_n, first_n_val, second_n, second_n_val):
    """ 
    Very specific where we fill @param first_n values of the array with the 
    @param first_n_val, and the @param second_n values of the array with 
    @param second_n_val.
    """
    ans = []
    for a in range(first_n):
        ans.append(first_n_val)
    
    for a in range(second_n):
        ans.append(second_n_val)
    
    return ans
    

def gen_exact_evid_array(sens, spans, data, conv):
    """ 
    Generate an array of length s that tells us which tokens are evidence, and which are not.
    """
    lengths = [len(sen[0]) for sen in sens]
    last = 0
    ans = []
    
    i = 0
    for l in lengths:
        st, end = last, last + l
        # we break if we find a evidence span that tells us that this sentence is evidence
        for start_span, end_span in spans:
            start_span -= 1
            is_evid = start_span <= st < end_span or start_span <= st < end_span
            if (is_evid):
                break
            
        # capture different senarios 
        if start_span <= st < end_span:
            """
            40 <= 60 < 80... 
            First (80 - 60) = 20 of the span good
            """
            first_n = min(end_span - st, l)
            left_over = l - first_n   
            ans.append(gen_first_n_second_n(first_n, 1, left_over, 0))
            
        elif start_span <= end < end_span:
            """
            40 <= 50 < 80
            The last n 
            """
            last_n  = end - start_span
            first_n = l - last_n
            ans.append(gen_first_n_second_n(first_n, 0, last_n, 1))       
        elif st <= start_span < end:
            """
            in the case that this sentence is bigger than the span...
            """
            first_n = start_span - st
            last_n = l - first_n
            ans.append(gen_first_n_second_n(first_n, 0, last_n, 1))
        else:
            ans.append([0] * l)
            
         # make sure that this isn't evidence...
        if not(1 in ans[-1]):
            data['sentence_span'][i][1] = 0
        
        last += l
        i += 1
        
    return ans

def find_span_location(sentences, starts, ends):
    """
    Find the sentence(s) that fall between the start and end integers.
    
    @param sentences is an array of [sentence_start, sentence_end, sentences] 
    for each sentence in the article, where sentence_start/end are the character offsets.
    @param start     is where the evidence-span(s) starts.
    @param end       is where the evidence-span(s) ends. 
    @param seq_to_str is a function that turns strings to arrays of data.
    
    """
    ans = []
    for s in sentences: 
        
        # check to see if this sentence is marked as evidence by any annotators
        is_evid = False
        for i in range(len(starts)):
            st = starts[i]
            ed = ends[i]
            
            # check if start is between our goals 
            is_evid = is_evid or (st <= s[0] and s[0] < ed) or (st <= s[1] and s[1] < ed)
            if (is_evid):
                break
        
        if (is_evid):
            ans.append([s[2], 1])
        else:
            ans.append([s[2], 0])
                    
    return ans

# spacy (Sentence tokenizer)
def split(text):
    """ Splits the text into sentences """
    doc = nlp(text)
    sentences = [x.text_with_ws for x in doc.sents]
    return sentences

def split_into_sentences(id_, text, sentence_splits):
    """ 
    Split the text into sentences in the form [[x, y, sentence]],
    where x is the start index in text, and y is the end index.
    
    @param id_ is the id for this article (i.e. PMC + id_ + .nxml)
    @param text is the text for the id_.
    @param sentence_split is a dictionary of id_ to outputs (so we don't repeat). 
=    """
    if id_ in sentence_splits: 
        return sentence_splits[id_]
    
    sentences = split(text) # grab sentences
    splits = [] # our return array
    last = 0 # the index of the last sentence
    
    for sen in sentences:
        splits.append([last, last + len(sen), sen])
        last = last + len(sen) # open-closed intervals
        
    sentence_splits[id_] = splits # store for future use
    return splits
