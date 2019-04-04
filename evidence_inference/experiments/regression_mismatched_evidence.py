# -*- coding: utf-8 -*-
"""
@author: Eric

This is a logistic regression model using SGD and binary cross entropy.
"""

import random
from os.path import join, dirname, abspath
import sys

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import copy
import torch
import numpy as np
import pandas as pd
import evidence_inference.preprocess.preprocessor as preprocessor
from evidence_inference.models.regression import train_model, test_model, bag_of_words
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

PROMPT_ID_COL_NAME = "PromptID"
LBL_COL_NAME = "Label Code"
EVIDENCE_COL_NAME = "Annotations"
STUDY_ID_COL = "PMCID"

USE_CUDA = False


def parse_prompt_id_data(annotations, prompts, pids, i):
    """ 
    Parse a single prompt and get the information necessary.
    
    @param annotations is the annotation dictionary that contains all answers/reasoning data.
    @param prompts     is the prompt dictionary that contains all i/c/o data.
    @param pids        is the ordering of the PMC id. 
    @param i           is which prompt to look at.
    
    @return id_        the prompt idenitification number.
    @return tmp        an arrary containing text, outcome, intervention, comparator.
    @return loss       a single value of -1/0/1 for sig. inc/ no diff. / sig. dec 
    """
    # information for this article
    annotations_for_prompt = annotations[annotations[PROMPT_ID_COL_NAME] == prompts["PromptID"].values[i]]
    labels = annotations_for_prompt[[LBL_COL_NAME, EVIDENCE_COL_NAME]].values
    id_ = pids[i]
    article = preprocessor.get_article(id_)
    article_text = preprocessor.extract_raw_text(article).lower()

    # extract i/c/o
    out = prompts["Outcome"].values[i].lower()
    inter = prompts["Intervention"].values[i].lower()
    cmp = prompts["Comparator"].values[i].lower()
    
    # this is all of the reasonings
    res = [a[1] for a in labels]
    
    losses = []
    data   = []
    
    # match up reasonings with text
    for r in res:    
        tmp = [article_text, r, out, inter, cmp]
        loss = stats.mode([l1[0] for l1 in labels])[0][0]
        
        data.append(tmp)
        losses.append(loss)
    
    return id_, data, losses


# noinspection PyUnresolvedReferences
def load_data(use_test, bow=True):
    """
    Load the data into a train/val/test set that allows for easy access.

    @return bag-of-word representation of training, validation, test sets (with labels).
    """

    prompts = preprocessor.read_prompts()
    annotations = preprocessor.read_annotations()

    # filter out prompts for which we do not have annotations for whatever reason
    # this was actually just one case; not sure what was going on there.
    def have_annotations_for_prompt(prompt_id):
        return len(annotations[annotations[PROMPT_ID_COL_NAME] == prompt_id]) > 0

    prompts = [prompt for row_idx, prompt in prompts.iterrows() if
               have_annotations_for_prompt(prompt[PROMPT_ID_COL_NAME])]
    prompts = pd.DataFrame(prompts)

    # Sort into training and validation by article id
    train_doc_ids = preprocessor.train_document_ids()
    val_doc_ids = preprocessor.validation_document_ids()
    test_doc_ids = preprocessor.test_document_ids()
    
    # get a dev set randomly
    dev_doc_ids = list(train_doc_ids)
    random.shuffle(dev_doc_ids)
    dev_doc_ids = set(dev_doc_ids[:int(len(dev_doc_ids) * .1)])

    x_train, y_train, x_dev, y_dev, x_val, y_val, x_test, y_test = [], [], [], [], [], [], [], []
    pids = prompts[STUDY_ID_COL].values
    for i in range(len(pids)):
        id_, data, losses = parse_prompt_id_data(annotations, prompts, pids, i)
        
        in_training = id_ in (train_doc_ids - dev_doc_ids) 
        # get a reasoning from previous/next prompt id
        if (i > 0 and id_ == pids[i - 1] and not(in_training)):
            _, mismatched_data, _ = parse_prompt_id_data(annotations, prompts, pids, i - 1)
            # add the mismatched data here
            row = copy.deepcopy(data[0])
            row[1] = mismatched_data[0][1]
            data.append(row)
            losses.append(losses[-1])

        elif (i < len(pids) and id_ == pids[i + 1] and not(in_training)):
            _, mismatched_data, _ = parse_prompt_id_data(annotations, prompts, pids, i + 1)
            # add the mismatched data here
            row = copy.deepcopy(data[0])
            row[1] = mismatched_data[0][1]
            data.append(row)
            losses.append(losses[-1])
        
        for i in range(len(data)):
            tmp  = data[i]
            loss = losses[i]
            
            # find where to put this section        
            if id_ in dev_doc_ids and not(use_test):
                x_dev.append(tmp)
                y_dev.append(loss)            
            elif id_ in train_doc_ids:
                x_train.append(tmp)
                y_train.append(loss)
            elif id_ in val_doc_ids:
                x_val.append(tmp)
                y_val.append(loss)
            elif id_ in test_doc_ids:
                x_test.append(tmp)
                y_test.append(loss)
            else:
                raise ValueError("Unknown study id {}".format(id_))
            
    # if we are removing the test set, use validation as test set.
    if not(use_test):
        x_test = x_val
        y_test = y_val
        x_val = x_dev
        y_val = y_dev
                
    ret = bag_of_words(x_train, y_train, x_val, y_val, x_test, y_test, 5) if bow else [x_train, y_train, x_val, y_val,
                                                                                    x_test, y_test]
    return ret


def main(iterations, use_test = False):
    """
    @param iterations is the number of epochs that the model runs for.
    @return accuracy, guessing accuracy, F1, guessing F1
    """
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(use_test)
    y_test += 1
    print("Loaded {} training examples, {} validation examples, {} testing examples".format(len(x_train), len(x_val), len(x_test)))
    model = train_model(x_train, y_train, x_val, y_val, iterations, learning_rate=0.001)
    preds = test_model(model, x_test)

    # calculate f1 and accuracy
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')

    # calculate the majority class f1 and accuracy
    mode = stats.mode(y_train)[0][0][0]
    majority_guess = [mode for _ in preds]
    guess_acc = accuracy_score(y_test, majority_guess)
    guess_f1 = f1_score(y_test, majority_guess, average='macro')

    return acc, f1, guess_acc, guess_f1


if __name__ == '__main__':
    if USE_CUDA:
        print("Cuda device number: {}, name: {}".format(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
    
    np.random.seed(500)
    random.seed(500)
    torch.manual_seed(500)
    
    i = 500
    acc, f1, acc_g, f1_g = main(i, True)
    print("Final ACC: {}".format(acc))
    print("Final guessing ACC: {}".format(acc_g))
    print("Final F1: {}".format(f1))
    print("Final guessing F1: {}".format(f1_g))
