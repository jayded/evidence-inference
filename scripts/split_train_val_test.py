# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:15:51 2018

@author: Eric
"""
import glob
import random
import numpy as np


# the locations of the files
loc = '/home/ubuntu/evidence-inference/annotations/'
#loc = '..//.//annotations//'
loc_files = loc + 'xml_files/*.nxml'
loc_train = loc + 'splits/' + 'train_article_ids.txt'
loc_val   = loc + 'splits/' + 'validation_article_ids.txt'
loc_test  = loc + 'splits/' + 'test_article_ids.txt'

# load in the articles
files = set(list(map(lambda x: int(x.split("\\")[-1].split(".")[0].split("PMC")[1]), glob.glob(loc_files))))

# load in the data
train = np.loadtxt(loc_train, dtype = int, delimiter = " ")
val   = np.loadtxt(loc_val, dtype = int, delimiter = " ")
test  = np.loadtxt(loc_test, dtype = int, delimiter = " ")

# define what we have already done
placed = set(train).union(set(val)).union(set(test))

# what needs to be placed
to_do = list(files - placed)

# Shuffle randomly, and split into 80, 10, 10 splits
random.shuffle(to_do)
tr_split = int(.8 * len(to_do)) # where to end
va_split = int(.9 * len(to_do)) # where to end
to_train = to_do[:tr_split]
to_val   = to_do[tr_split:va_split]
to_test  = to_do[va_split:]

# Notify user of progress
print("Training articles   added: {}".format(len(to_train))) 
print("Validation articles added: {}".format(len(to_val))) 
print("Testing articles    added: {}".format(len(to_test))) 

# append to existing datasets
train = np.append(train, to_train)
val   = np.append(val,   to_val)
test  = np.append(test,  to_test)

# save the new files
np.savetxt(loc_train, train, fmt = "%d", delimiter = " ")
np.savetxt(loc_val,   val,   fmt = "%d", delimiter = " ")
np.savetxt(loc_test,  test,  fmt = "%d", delimiter = " ")

# Notify user of progress
print("Final training   articles: {}".format(len(train))) 
print("Final validation articles: {}".format(len(val))) 
print("Final testing    articles: {}".format(len(test))) 