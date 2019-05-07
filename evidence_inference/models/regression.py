# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:59:23 2018

@author: Eric

This is a logistic regression model using SGD and binary cross entropy.
"""
from os.path import join, dirname, abspath
import sys

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import copy
import torch
import random
import numpy as np
import pandas as pd
import evidence_inference.preprocess.preprocessor as preprocessor
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer

PROMPT_ID_COL_NAME = "PromptID"
LBL_COL_NAME = "Label Code"
EVIDENCE_COL_NAME = "Annotations"
STUDY_ID_COL = "PMCID"

USE_CUDA = False

def flatten(x):
    """
    Turns 2D array into 1D.
    """
    return np.asarray(x).reshape(len(x) * np.asarray(x).shape[1])


# noinspection PyCallingNonCallable
def fmt_n(x, n = 4):
    """
    Take a flatten 1D array and pair it into groups of n.
    """
    if USE_CUDA:
        return torch.tensor(x.reshape((int(len(x) / n), n * x.shape[1])), dtype=torch.float32).cuda()
    else:
        return torch.tensor(x.reshape((int(len(x) / n), n * x.shape[1])), dtype=torch.float32)


# noinspection PyCallingNonCallable
def bag_of_words(x_train, y_train, x_val, y_val, x_test, y_test, n = 4):
    """
    Transform data into b.o.w. w/ a max of 20k tokens.

    @param x_train is the training data.
    @param y_train is the training labels.
    @param x_val   is the validation data.
    @param y_val   is the validation labels.
    @param x_test  is the training data.
    @param y_test  is the test labels.
    @param n       is the number of sections in the input data (text/ico/reasoning).
    @return a bag of words representation of the data.
    """
    x_train, x_val, x_test = flatten(x_train), flatten(x_val), flatten(x_test)
    print("flatten")
    vectorizer = CountVectorizer(max_features=20000)
    X = vectorizer.fit_transform(x_train)
    print("trained")
    y_train, y_val, y_test = (torch.tensor([[y] for y in x], dtype=torch.long).cuda() if USE_CUDA else torch.tensor([[y] for y in x], dtype=torch.long) for x in (np.asarray(y_train), np.asarray(y_val), np.asarray(y_test)))
    print("y_s achieved.")
    return fmt_n(X.toarray(), n), y_train, fmt_n(vectorizer.transform(x_val).toarray(), n), y_val, fmt_n(vectorizer.transform(x_test).toarray(), n), y_test


# noinspection PyUnresolvedReferences
def load_data(use_test, bow=True):
    """
    Load the data into a train/val/test set that allows for easy access.

    @return bag-of-word representation of training, validation, test sets (with labels).
    """
    print("Loading data.")
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
        annotations_for_prompt = annotations[annotations[PROMPT_ID_COL_NAME] == prompts["PromptID"].values[i]]
        labels = annotations_for_prompt[[LBL_COL_NAME, EVIDENCE_COL_NAME]].values
        id_ = pids[i]
        article = preprocessor.get_article(id_)
        article_text = preprocessor.extract_raw_text(article).lower()

        # extract i/c/o
        out = prompts["Outcome"].values[i].lower()
        inter = prompts["Intervention"].values[i].lower()
        cmp = prompts["Comparator"].values[i].lower()

        # add to correct pile: train/val/test
        tmp = [article_text, out, inter, cmp]
        loss = stats.mode([l1[0] for l1 in labels])[0][0]
        
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
     
    print("Generating bag of words...")
    ret = bag_of_words(x_train, y_train, x_val, y_val, x_test, y_test) if bow else [x_train, y_train, x_val, y_val,
                                                                                    x_test, y_test]
    return ret


# noinspection PyCallingNonCallable
def train_model(x_train, y_train_e, x_val, y_val_e, num_epochs, learning_rate):
    """
    Train the model using the data given, along with the parameters given.

    @param x_train       is the training dataset.
    @param y_train_e     is an np.array of the training labels.
    @param x_val         is the validation set.
    @param y_val_e       is the np.array of validation labels.
    @param num_epochs    for the model to use.
    @param learning_rate for the model to use.
    @return              the best trained model thus far (based on validation accuracy).
    """
    y_train_e += 1
    y_val_e += 1

    # Logistic regression model
    model = nn.Linear(x_val.shape[1], 3)
    model.float()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    if USE_CUDA:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    best_model = None
    best_accuracy = float('-inf')
    # Training phase
    for epoch in range(num_epochs):
        for i in range(len(x_train)):

            # Forward pass
            tmp = x_train[i].reshape(-1, len(x_train[i]))
            output = model(tmp)
            loss = criterion(output, y_train_e[i])
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check if this is the best model thus far?
        with torch.no_grad():
            preds = test_model(model, x_val)
            
        preds = preds
        y_val_e = y_val_e.cpu()
        acc = accuracy_score(y_val_e, preds)
        if acc > best_accuracy:
            print("Best acc: {:.3f}".format(acc))
            best_accuracy = acc
            best_model = copy.deepcopy(model)

        print("Epoch: {}, validation accuracy: {:.3f}".format(epoch + 1, acc))

    return best_model


# noinspection PyUnresolvedReferences,PyCallingNonCallable
def test_model(model, x_test):
    """
    Get a prediction out of the model and return it.

    @param model is a pytorch model.
    @param x_test is a numpy.array that will be turned into a tensor for prediction.
    @return a set of predictions.
    """
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted.cpu().numpy()


def main(iterations, use_test):
    """
    The main method that trains and test the model using a training, validation, and
    test set.

    @param iterations is the number of epochs that the model runs for.
    @return accuracy, guessing accuracy, F1, guessing F1
    """
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(use_test)
    y_test += 1
    print("Loaded {} training examples, {} validation examples, {} testing examples".format(len(x_train), len(x_val), len(x_test)))
    model = train_model(x_train, y_train, x_val, y_val, iterations, learning_rate=0.001)
    preds = test_model(model, x_test)
    
    y_test = y_test.cpu()
    # calculate f1 and accuracy
    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds, average='macro')
    prec = precision_score(y_test, preds, average = 'macro')
    rec  = recall_score(y_test, preds, average = 'macro')

    # calculate the majority class f1 and accuracy
    mode = stats.mode(y_train.cpu())[0][0][0]
    majority_guess = [mode for _ in preds]
    guess_acc = accuracy_score(y_test, majority_guess)
    guess_f1 = f1_score(y_test, majority_guess, average='macro')
    guess_prec = precision_score(y_test, majority_guess, average = 'macro')
    guess_rec  = recall_score(y_test, majority_guess, average = 'macro')

    return acc, f1, prec, rec, guess_acc, guess_f1, guess_prec, guess_rec


if __name__ == '__main__':
    #print("Cuda device number: {}, name: {}".format(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
    np.random.seed(500)
    random.seed(500)
    torch.manual_seed(500)

    i = 25
    for j in range(5):
        acc, f1, prec, rec, acc_g, f1_g, prec_g, rec_g = main(i, True)
        print("Final ACC: {}".format(acc))
        print("Final guessing ACC: {}".format(acc_g))
        print("Final F1: {}".format(f1))
        print("Final guessing F1: {}".format(f1_g))
        print("Final Precision: {}".format(prec))
        print("Final guessing Precision: {}".format(prec_g))
        print("Final Recall: {}".format(rec))
        print("Final guessing Recall: {}".format(rec_g))
