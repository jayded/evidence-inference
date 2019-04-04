# -*- coding: utf-8 -*-
import re
from string import punctuation
from functools import reduce

import spacy
nlp = spacy.load('en_core_web_sm')


"""
Removes the punctuation from the given string s.

@param str s is the string to be stripped of punctuation.
"""
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
    
"""
Determines if s is an integer

@param str s is the string that potentially is an integer.
"""    
def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
        
        
"""
Finds a p-value through iterating until it finds a string value. Will return -1
if there is an error.

@param str pval is of the form "0.05 is a good number."
@return an integer representing the first number found.
"""
def parse_p_value(pval):
    val = ""
    i = 0
    pval = pval.lstrip() # this gets rid of white space in the beginning.
    
    while (i < len(pval) and isInt(val + pval[i])):
        val += pval[i]
        i += 1
        
    if (val == ""):
        return -1
    else:
        return int(val)
        
        
"""
Find each sentence and split it up.

@param str text - the paragraph to split up.
@return an array of sentences.
"""
def split_sentences(text):
    # spacy (Sentence tokenizer)
    def split(text):
        """ Splits the text into sentences """
        doc = nlp(text)
        sentences = [x.text_with_ws for x in doc.sents]
        return sentences
    
    return split(text)
        

"""
Re-calculating the likelihood of non-significance and significance.

@param str sen the likely sentence to contain the answer.
@param str find i.e. (p =, p=, p<, etc.)
@param int loc_word is the location of the word in sen (usually cmp/inter).
@param int loc is an array of the locations in which the "find" is in "sen."
@param int ns is the current points awarded for non-significance.
@param int sig is the current points awarded for significance.
"""        
def sig_or_ns_pval(sen, find, loc_word, loc, ns, sig):
    closest_p_value = reduce((lambda y, x: y if (abs(loc_word - x) > y) else x), loc, -1)
    if (closest_p_value > 0):            
        pval = parse_p_value(sen[closest_p_value + len(find):])   
    else:
        pval = -1
        
    if (pval >= 0.05 and pval > 0):
        ns += 1
    elif (pval > 0): 
        sig += 1

    return ns, sig