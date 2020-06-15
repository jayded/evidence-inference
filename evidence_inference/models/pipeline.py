import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
import sys

from collections import Counter, defaultdict, OrderedDict
from dataclasses import asdict, dataclass
from os.path import join, dirname, abspath
from typing import Callable, Dict, List, Set, Tuple, Union

import numpy as np
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

from dacite import from_dict
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from transformers import BertTokenizer

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

from evidence_inference.preprocess import preprocessor
from evidence_inference.preprocess.preprocessor import (
    PROMPT_ID_COL_NAME,
    LABEL,
    EVIDENCE_COL_NAME,
    EVIDENCE_START,
    EVIDENCE_END,
    STUDY_ID_COL,
)
from evidence_inference.preprocess.representations import (
    Document,
    Sentence,
    Token,
    to_structured,
    retokenize_with_bert
)
from evidence_inference.models.bert_model import initialize_models

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)

# TODO it would be nice to refreeze this, but there's a poor interaction between immutability, default arguments, and __post_init__ processing
@dataclass(frozen=False, repr=True, eq=True)
class Annotation:
    doc: Document
    prompt_id: str
    tokenized_sentences: List[List[int]]
    i: Union[str, Tuple[int, ...], torch.IntTensor]
    c: Union[str, Tuple[int, ...], torch.IntTensor]
    o: Union[str, Tuple[int, ...], torch.IntTensor]
    evidence_texts: Tuple[Union[str, List[int]]]
    evidence_spans: Tuple[Tuple[int, int], ...]
    evidence_vector: Union[Tuple[int, ...], torch.IntTensor]
    significance_class: Union[str, int]

    #def __post_init__(self, i, c, o, evidence_texts, evidence_spans, evidence_vector):
    #    def tupleify(obj):
    #        if isinstance(obj, (list, iterable)):
    #            obj = tuple(obj)
    #        else:
    #            assert isinstance(obj, (int, str, tuple))
    #        return obj
    #    self.i = tupleify(i)
    #    self.c = tupleify(c)
    #    self.o = tupleify(o)
    #    self.evidence_texts = tupleify(evidence_texts)
    #    self.evidence_spans = tupleify(evidence_spans)
    #    assert isinstance(evidence_vector, (tuple, list))
    #    self.evidence_vector = tupleify(evidence_vector)

    def retokenize(self, bert_tokenizer, do_intern=True) -> 'Annotation':
        def handle_str(s):
            if isinstance(s, str):
                if do_intern:
                    return bert_tokenizer.encode(s, add_special_tokens=False)
                else:
                    return bert_tokenizer.tokenize(s, add_special_tokens=False)
            elif isinstance(s, (list, tuple)) and do_intern:
                ret = []
                for elem in s:
                    ret.extend(handle_str(elem))
                return ret
            else:
                raise ValueError(f'Attempted to retokenize or tokenize untokenizeable instance {s}')
        
        return Annotation(doc=self.doc,
                          prompt_id=self.prompt_id,
                          tokenized_sentences=self.tokenized_sentences,
                          i=tuple(handle_str(self.i)),
                          c=tuple(handle_str(self.c)),
                          o=tuple(handle_str(self.o)),
                          evidence_texts=tuple(handle_str(s) for s in set(map(str.lower, filter(lambda x: isinstance(x, str), self.evidence_texts)))),
                          evidence_spans=self.evidence_spans,
                          evidence_vector=self.evidence_vector,
                          significance_class=self.significance_class)

def get_identifier_sampler(params: dict) -> Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]]:
    if 'length_ratio' in params['evidence_identifier']:
        return get_length_identifier_sampler(params)
    ratio = params['evidence_identifier']['sampling_ratio']
    # returns sentence text, ico, classification
    def identifier_sampler(ann: Annotation) -> List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]:
        pos = []
        neg = []
        for tokens, sent in zip(ann.tokenized_sentences, ann.doc.sentences):
            i = torch.IntTensor(ann.i)
            c = torch.IntTensor(ann.c)
            o = torch.IntTensor(ann.o)
            if sent.labels is not None and sent.labels['evidence'] == 1:
                pos.append((tokens, (i, c, o), 1))
            else:
                neg.append((tokens, (i, c, o), 0))
        samples = random.sample(neg, k=min(len(neg), int(ratio * len(pos)))) + pos
        random.shuffle(samples)
        return samples
    return identifier_sampler

def get_length_identifier_sampler(params: dict) -> Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]]:
    ratio = params['evidence_identifier']['sampling_ratio']
    length_ratio = params['evidence_identifier']['length_ratio']
    # returns sentence text, ico, classification
    def identifier_sampler(ann: Annotation) -> List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]:
        pos = []
        neg = []
        for tokens, sent in zip(ann.tokenized_sentences, ann.doc.sentences):
            i = torch.IntTensor(ann.i)
            c = torch.IntTensor(ann.c)
            o = torch.IntTensor(ann.o)
            if sent.labels is not None and sent.labels['evidence'] == 1:
                pos.append((tokens, (i, c, o), 1))
            else:
                neg.append((tokens, (i, c, o), 0))
        #samples = random.sample(neg, k=min(len(neg), int(ratio * len(pos)))) + pos
        #random.shuffle(samples)
        #return samples
        samples = list(pos)
        for p in pos:
            lower_bound = len(pos) * (1 - length_ratio)
            upper_bound = len(pos) * (1 + length_ratio)
            acceptable_negatives = list(filter(lambda n: lower_bound <= len(n[0]) and upper_bound >= len(n[0]), neg))
            if len(acceptable_negatives) > ratio:
                samples.extend(random.sample(acceptable_negatives, k=math.ceil(ratio)))
            else:
                samples.extend(acceptable_negatives)
        random.shuffle(samples)
        return samples
    return identifier_sampler

def identifier_everything_sampler(ann: Annotation) -> List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]:
    ret = []
    for tokens, sent in zip(ann.tokenized_sentences, ann.doc.sentences):
        i = torch.IntTensor(ann.i)
        c = torch.IntTensor(ann.c)
        o = torch.IntTensor(ann.o)
        if sent.labels is not None and sent.labels['evidence'] == 1:
            cls = 1
        else:
            cls = 0
        ret.append((tokens, (i, c, o), cls))
    return ret


def get_classifier_oracle_sampler(params: dict) -> Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]]:
    evidence_classes = params['evidence_classifier']['classes']
    ec_map = {x:i for i,x in enumerate(evidence_classes)}
    def classifier_oracle_sampler(ann: Annotation) -> List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]:
        pos = []
        for tokens, sent in zip(ann.tokenized_sentences, ann.doc.sentences):
            i = torch.IntTensor(ann.i)
            c = torch.IntTensor(ann.c)
            o = torch.IntTensor(ann.o)
            if sent.labels is not None and sent.labels['evidence'] == 1:
                pos.append((tokens, (i, c, o), ann.significance_class))
        if len(pos) == 0:
            return pos
        return random.sample(pos, k=1)
    return classifier_oracle_sampler

def mask_tokens(sampler: Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]],
                pad_token_id: int) -> Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]]:
    def masked_sampler(ann: Annotation):
        samples = sampler(ann)
        ret = []
        for _, ico, cls in samples:
            ret.append((torch.IntTensor([pad_token_id]), ico, cls))
        return ret
    return masked_sampler

def get_classifier_sampler(params: dict) -> Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]]:
    evidence_classes = params['evidence_classifier']['classes']
    ec_map = {x:i for i,x in enumerate(evidence_classes)}
    def classification_sampler(ann: Annotation) -> List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]:
        ret = []
        for tokens, sent in zip(ann.tokenized_sentences, ann.doc.sentences):
            i = torch.IntTensor(ann.i)
            c = torch.IntTensor(ann.c)
            o = torch.IntTensor(ann.o)
            if sent.labels is not None and sent.labels['evidence'] == 1:
                ret.append((tokens, (i, c, o), ann.significance_class))
        random.shuffle(ret)
        return ret 
    return classification_sampler

def classifier_everything_sampler(ann: Annotation) -> List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]:
    ret = []
    for tokens, sent in zip(ann.tokenized_sentences, ann.doc.sentences):
        i = torch.IntTensor(ann.i)
        c = torch.IntTensor(ann.c)
        o = torch.IntTensor(ann.o)
        ret.append((tokens, (i, c, o), ann.significance_class))
    return ret 

def mask_sampler(sampler: Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]],
                 fields_to_mask: Set[str]) -> Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]]:
    fields_to_tuple_pos = {
        'i': 0,
        'c': 1,
        'o': 2,
    }
    fields_to_mask = {fields_to_tuple_pos[f] for f in fields_to_mask}
    pad = torch.IntTensor([0])
    def get_sampler(ann: Annotation) -> List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]:
        unmasked = sampler(ann)
        ret = []
        for (sent, ico, kls) in unmasked:
            new_ico = []
            for i, ico_elem in enumerate(ico):
                if i in fields_to_mask:
                    new_ico.append(pad)
                else:
                    new_ico.append(ico_elem)
            ret.append((sent, tuple(new_ico), kls))
        return ret
    return get_sampler

def make_preds_batch(classifier: nn.Module,
                     batch_elements: List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]],
                     sep_token_id: int,
                     device: str=None,
                     criterion: nn.Module=None) -> Tuple[float, List[float], List[int], List[int]]:
    """Batch predictions

    Args:
        classifier: a module that looks like an AttentiveClassifier
        batch_elements: a list of elements to make predictions over. These must be SentenceEvidence objects.
        device: Optional; what compute device this should run on
        criterion: Optional; a loss function
    """
    # delete any "None" padding, if any (imposed by the use of the "grouper")
    sentences, icos, targets = zip(*filter(lambda x: x, batch_elements))
    targets = torch.tensor(targets, dtype=torch.long)
    sep = torch.tensor([sep_token_id], dtype=torch.int)
    queries = [torch.cat([i, sep, c, sep, o]).to(dtype=torch.long) for (i,c,o) in icos]
    sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]
    preds = classifier(queries, sentences)
    targets = targets.to(device=preds.device)
    if criterion:
        loss = criterion(preds, targets)
    else:
        loss = None
    preds = F.softmax(preds).cpu()
    hard_preds = torch.argmax(preds, dim=-1).cpu()
    return loss, preds, hard_preds, targets.cpu()

def make_preds_epoch(classifier: nn.Module,
                     data: List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]],
                     sep_token_id: int,
                     batch_size: int,
                     device: str=None,
                     criterion: nn.Module=None):
    """Predictions for more than one batch.

    Args:
        classifier: a module that looks like an AttentiveClassifier
        data: a list of elements to make predictions over. These must be SentenceEvidence objects.
        batch_size: the biggest chunk we can fit in one batch.
        device: Optional; what compute device this should run on
        criterion: Optional; a loss function
    """
    assert len(data) > 0
    epoch_loss = 0
    epoch_soft_pred = []
    epoch_hard_pred = []
    epoch_truth = []
    batches = _grouper(data, batch_size)
    classifier.eval()
    for batch in batches:
        with torch.no_grad():
            loss, soft_preds, hard_preds, targets = make_preds_batch(classifier, batch, sep_token_id, device, criterion=criterion)
        if loss is not None:
            epoch_loss += loss.sum().item()
        else:
            epoch_loss = None
        epoch_hard_pred.extend(hard_preds)
        epoch_soft_pred.extend(soft_preds.numpy())
        epoch_truth.extend(targets.numpy())
    if epoch_loss is not None:
        epoch_loss /= len(data)
    epoch_hard_pred = [x.item() for x in epoch_hard_pred]
    epoch_truth = [x.item() for x in epoch_truth]
    return epoch_loss, epoch_soft_pred, epoch_hard_pred, epoch_truth

# copied from https://docs.python.org/3/library/itertools.html#itertools-recipes
def _grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

@dataclass(frozen=True, repr=True, eq=True)
class DecodeInstance:
    docid: str
    prompt_id: str
    idx: int
    sentence: torch.IntTensor
    ico: Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor]
    identifier_class: int
    classification_class: int

    def to_model_input(self, class_type) -> Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]:
        ico = self.ico
        sent = self.sentence
        if class_type == 'identifier':
            cls = self.identifier_class
        elif class_type == 'classifier':
            cls = self.classification_class
        elif class_type == 'unconditioned_classifier':
            cls = self.classification_class
            ico = (torch.IntTensor([0]), torch.IntTensor([0]), torch.IntTensor([0]))
        elif class_type == 'unconditioned_identifier':
            cls = self.identifier_class
            ico = (torch.IntTensor([0]), torch.IntTensor([0]), torch.IntTensor([0]))
        elif class_type == 'ico_only':
            cls = self.classification_class
            sent = (0,)
        else:
            raise ValueError(f'unknown type {class_type}')
        return (sent, ico, cls)

def locate_known_evidence_snippets(data: List[Annotation]) -> Dict[str, Dict[int, Set[str]]]:
    ret = defaultdict(lambda: defaultdict(set)) # docid -> {(prompt_id, sentence_idx)}
    for ann in data:
        for s, sent in enumerate(ann.doc.sentences):
            if sent.labels is not None and sent.labels.get('evidence', 0) == 1:
                ret[ann.doc.docid][s].add(ann.prompt_id)
    out = dict()
    for k,v in ret.items():
        out[k] = dict()
        for k2, v2 in v.items():
            out[k][k2] = v2
    return out

def oracle_decoding_instances(data: List[Annotation]) -> List[DecodeInstance]:
    # TODO this should shift from being "all" by default to being something configurable.
    oracle_instances = [] # entire oracle spans
    for ann in data:
        # note that the oracle here isn't masked!
        for ev_id, ev_text in enumerate(ann.evidence_texts):
            oracle_instances.append(DecodeInstance(docid=ann.doc.docid,
                                                   prompt_id=ann.prompt_id,
                                                   idx=-1 * ev_id,
                                                   sentence=ev_text,
                                                   ico=(torch.IntTensor(ann.i), torch.IntTensor(ann.c), torch.IntTensor(ann.o)),
                                                   identifier_class=1,
                                                   classification_class=ann.significance_class))
    return oracle_instances

def decoding_instances(data: List[Annotation], identifier_transform, classifier_transform) -> List[DecodeInstance]:
    instances = []
    for ann in data:
        id_data = identifier_transform(ann)
        class_data = classifier_transform(ann)
        for s, ((id_tokens, id_ico, id_kls), (cls_tokens, cls_ico, cls_kls)) in enumerate(zip(id_data, class_data)):
            assert torch.all(id_tokens == cls_tokens)
            assert all(torch.all(x == y) for (x,y) in zip(id_ico, cls_ico))
            instances.append(DecodeInstance(docid=ann.doc.docid,
                                            prompt_id=ann.prompt_id,
                                            idx=s,
                                            sentence=id_tokens,
                                            ico=id_ico,
                                            identifier_class=id_kls,
                                            classification_class=cls_kls))
    return instances

def e2e_score(tru, pred, name, evidence_classes):
    acc = accuracy_score(tru, pred)
    f1 = classification_report(tru, pred, output_dict=False, digits=4, target_names=evidence_classes)
    conf_matrix = confusion_matrix(tru, pred, normalize='true')
    logging.info(f'{name} classification accuracy {acc},\nf1:\n{f1}\nconfusion matrix:\n{conf_matrix}\n')

def decode(evidence_identifier: nn.Module,
           evidence_classifier: nn.Module,
           unconditioned_evidence_identifier: nn.Module,
           save_dir: str,
           data_name: str,
           data: List[Annotation],
           model_pars: dict,
           sep_token_id: int,
           identifier_transform: Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]],
           classifier_transform: Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]],
           detokenizer=None,
           conditioned: bool=True):
    bad_ids = set(['2206488'])
    data = list(filter(lambda x: x.doc.docid not in bad_ids, data))
    # TODO consider saving the input data as well!
    save_dir = os.path.join(save_dir, 'decode', data_name)
    os.makedirs(save_dir, exist_ok=True)
    instances_save_file = os.path.join(save_dir, f'{data_name}_instances_output.pkl')
    identifier_save_file = os.path.join(save_dir, f'{data_name}_identifier_output.pkl')
    classifier_save_file = os.path.join(save_dir, f'{data_name}_classifier_output.pkl')
    oracle_classifier_save_file = os.path.join(save_dir, f'{data_name}_oracle_classifier_output.pkl')
    unconditioned_identifier_file = os.path.join(save_dir, f'{data_name}_unconditioned_identifier_output.pkl')
    logging.info(f'Decoding {len(data)} documents from {data_name}')
    batch_size = model_pars['evidence_identifier']['batch_size']
    evidence_classes = model_pars['evidence_classifier']['classes']
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        if os.path.exists(instances_save_file):
            logging.info(f'Loading instances from {instances_save_file}')
            instances, oracle_instances = torch.load(instances_save_file)
        else:
            logging.info(f'Generating and saving instances to {instances_save_file}')
            oracle_instances = oracle_decoding_instances(data)
            instances = decoding_instances(data, identifier_transform, classifier_transform)
            torch.save((instances, oracle_instances), instances_save_file)
        logging.info(f'Have {len(instances)} instances in our data')
        evidence_identifier.eval()
        evidence_classifier.eval()
        #import ipdb; ipdb.set_trace()
        if os.path.exists(identifier_save_file):
            logging.info(f'Loading evidence identification predictions on {data_name} from {identifier_save_file}')
            id_loss, id_soft_pred, id_hard_pred, id_truth = torch.load(identifier_save_file)
        else:
            logging.info(f'Making evidence identification predictions on {data_name} and saving to {identifier_save_file}')
            decode_target = 'identifier' if conditioned else 'unconditioned_identifier'
            id_loss, id_soft_pred, id_hard_pred, id_truth = make_preds_epoch(evidence_identifier,
                                                                             [instance.to_model_input(decode_target) for instance in instances],
                                                                             sep_token_id,
                                                                             batch_size,
                                                                             device=next(evidence_identifier.parameters()).device,
                                                                             criterion=criterion)
            torch.save((id_loss, id_soft_pred, id_hard_pred, id_truth), identifier_save_file)

        if os.path.exists(classifier_save_file):
            logging.info(f'Loading evidence classification predictions on {data_name} from {classifier_save_file}')
            cls_loss, cls_soft_pred, cls_hard_pred, cls_truth = torch.load(classifier_save_file)
        else:
            logging.info(f'Making evidence classification predictions on {data_name} and saving to {classifier_save_file}')
            decode_target = 'classifier' if conditioned else 'unconditioned_classifier'
            cls_loss, cls_soft_pred, cls_hard_pred, cls_truth = make_preds_epoch(evidence_classifier, 
                                                                                 [instance.to_model_input(decode_target) for instance in instances],
                                                                                 sep_token_id,
                                                                                 batch_size,
                                                                                 device=next(evidence_classifier.parameters()).device,
                                                                                 criterion=criterion)
            torch.save((cls_loss, cls_soft_pred, cls_hard_pred, cls_truth), classifier_save_file)

        if os.path.exists(unconditioned_identifier_file) and unconditioned_evidence_identifier is not None:
            logging.info('Loading unconditioned evidence identification data')
            docid_to_evidence_snippets, unid_soft_pred, unid_hard_pred = torch.load(unconditioned_identifier_file)
        elif unconditioned_evidence_identifier is not None:
            logging.info('Computing unconditioned evidence identification data')
            docid_to_evidence_snippets = locate_known_evidence_snippets(data)
            _, unid_soft_pred, unid_hard_pred, _ = make_preds_epoch(unconditioned_evidence_identifier,
                                                                    [instance.to_model_input('unconditioned_identifier') for instance in instances],
                                                                    sep_token_id,
                                                                    batch_size,
                                                                    device=next(unconditioned_evidence_identifier.parameters()).device,
                                                                    criterion=None)
            torch.save([docid_to_evidence_snippets, unid_soft_pred, unid_hard_pred], unconditioned_identifier_file)
        else:
            docid_to_evidence_snippets = locate_known_evidence_snippets(data)
            unid_soft_pred, unid_hard_pred = None, None

        logging.info('Aggregating information for scoring')
        # not all annotations have evidence due to lossy offset recovery
        annotations_with_evidence = set()
        annotations_without_evidence = set()
        # (prompt, docid) -> (id_truth) for best evidence sentence
        top1_pipeline_id_truth = dict()
        top1_pipeline_unid_truth = dict()
        # (prompt, docid) -> cls prediction for best evidence sentence
        top1_pipeline_cls_prediction = dict()
        # (prompt, docid) -> cls prediction for *every* evidence sentence
        oracle_pipeline_predictions = defaultdict(list)
        # (prompt, docid) -> [soft predictions]
        all_id_predictions = dict()  # list of p(sent=1|ico)
        all_id_truths = dict()
        all_cls_predictions = dict()  # list of p(sig-|ico), p(no sig|ico), p(sig+|ico)
        all_unconditional_id_predictions = dict()  # list of p(sent=1)

        # (prompt, docid) -> (truth, prediction)
        correct_evidence_predictions = dict() # when we get a correct evidence, what does our classification look like?
        incorrect_evidence_predictions = dict() # when we get an incorrect evidence, what does our classification look like?
        no_evidence_predictions = dict() # when we can't know if we get an evidence, what does our classification look like?
        counterfactual_incorrect_evidence_predictions = dict() # when we made an incorrect evidence classification, what does the classification over a correct evidence look like?
        unconditioned_cls = dict() 

        # TODO how many of the identification mistakes are plausible?

        total_length = 0
        #import ipdb; ipdb.set_trace()
        for ann in data:
            doc_length = len(ann.tokenized_sentences)
            key = (ann.prompt_id, ann.doc.docid)
            id_predictions = id_soft_pred[total_length:total_length + doc_length]
            hard_id_predictions = id_hard_pred[total_length:total_length + doc_length]
            #unid_hard_predictions = unid_hard_pred[total_legnth:total_length + len(ann.tokenized_sentences)]
            id_truths = id_truth[total_length:total_length + doc_length]
            has_ev = sum(id_truths) > 0
            if has_ev:
                annotations_with_evidence.add(key)
            else:
                annotations_without_evidence.add(key)
            cls_predictions = cls_hard_pred[total_length:total_length + doc_length]

            best_id_sent_idx = np.argmax([id_pred[1] for id_pred in id_predictions])
            id_tru = id_truth[total_length + best_id_sent_idx]
            cls_tru = cls_truth[total_length + best_id_sent_idx]
            cls_pre = cls_hard_pred[total_length + best_id_sent_idx]

            top1_pipeline_id_truth[key] = id_tru
            top1_pipeline_cls_prediction[key] = (cls_tru, cls_pre)
            if id_tru == 1:
                correct_evidence_predictions[key] = (cls_tru, cls_pre)
            elif has_ev:
                incorrect_evidence_predictions[key] = (cls_tru, cls_pre)
            else:
                no_evidence_predictions[key] = (cls_tru, cls_pre)

            assert len(id_truths) == len(hard_id_predictions)
            for (id_sent_truth, cls_pred) in zip(id_truths, cls_predictions):
                if id_sent_truth == 1:
                    oracle_pipeline_predictions[key].append((ann.significance_class, cls_pred))

            all_id_predictions[key] = [id_pred[1] for id_pred in id_predictions]
            all_id_truths[key] = id_truths
            all_cls_predictions[key] = cls_predictions

            if unid_soft_pred:
                unid_soft_predictions = unid_soft_pred[total_length:total_length + doc_length]
                all_unconditional_id_predictions[key] = [id_pred[1] for id_pred in unid_soft_predictions]
                best_unid_sent_idx = np.argmax(all_unconditional_id_predictions[key])
                top1_pipeline_unid_truth[key] = id_truth[total_length + best_unid_sent_idx]
                unconditioned_cls[key] = (cls_truth[total_length + best_unid_sent_idx], np.argmax(cls_predictions[best_unid_sent_idx]))
            total_length += len(ann.tokenized_sentences)

        assert total_length == len(cls_hard_pred)
        assert total_length == len(id_hard_pred)
        logging.info(f'Of {len(data)} annotations, {len(annotations_with_evidence)} have evidence spans, {len(annotations_without_evidence)} do not')
        
        def id_scores(id_tru, id_soft_preds, top1_values, top1_cls_preds=None, preds_dict=None, preds_truth=None):
            # auc for id prediction
            id_auc = roc_auc_score(id_tru, id_soft_preds)
            # accuracy for top 1 per document
            id_top1_acc = accuracy_score([1] * len(top1_values), list(top1_values))
            # accuracy for all documents/sengences
            id_all_acc = accuracy_score(id_tru, [round(x) for x in id_soft_preds]) # TODO this should just be passed in instead of rounding...
            if preds_dict is not None:
                assert preds_truth is not None
                mrr = 0
                query_count = 0
                for k in preds_truth.keys():
                    pt = zip(preds_dict[k], preds_truth[k])
                    pt = sorted(pt, key=lambda x: x[0], reverse=True)
                    for pos, (_, t) in enumerate(pt):
                        if t == 1:
                            mrr += 1 / (1 + pos)
                            query_count += 1
                            break
                mrr = mrr / query_count
            else:
                mrr = None
            logging.info(f'identification auc {id_auc}, top1 acc: {id_top1_acc}, everything acc: {id_all_acc}, mrr: {mrr}')

            if top1_cls_preds is not None:
                assert preds_dict is not None
                mistakes = defaultdict(lambda: defaultdict(lambda: 0))
                for eyeD, (cls_tru, _) in top1_cls_preds.items():
                    pt = zip(preds_dict[eyeD], preds_truth[eyeD])
                    pt = sorted(pt, key=lambda x: x[0], reverse=True)
                    if pt[0][1] == 1:
                        mistakes[cls_tru]['tp'] += 1
                    else:
                        mistakes[cls_tru]['fp'] += 1
                mistakes_str = []
                for tru, preds in mistakes.items():
                    tp, fp = preds['tp'], preds['fp']
                    frac = tp / (tp + fp)
                    mistakes_str.append(f'cls {tru} evid accuracy {frac}')
                mistakes_str = '  '.join(mistakes_str)
                logging.info(f'Evidence ID accuracy breakdown by classification types {mistakes_str}')
        
        ev_only_id_soft_pred = list(itertools.chain.from_iterable(all_id_predictions[x] for x in annotations_with_evidence))
        ev_only_id_truth = list(itertools.chain.from_iterable(all_id_truths[x] for x in annotations_with_evidence))
        id_scores(ev_only_id_truth, ev_only_id_soft_pred, [top1_pipeline_id_truth[x] for x in annotations_with_evidence], top1_cls_preds=top1_pipeline_cls_prediction, preds_dict=all_id_predictions, preds_truth=all_id_truths)
        #id_scores(id_truth, id_soft_pred, top1_pipeline_id_truth.values())

        # pipeline score
        pipeline_truth, pipeline_pred = zip(*top1_pipeline_cls_prediction.values())
        e2e_score(pipeline_truth, pipeline_pred, 'Pipeline', evidence_classes)

        # oracle classification F1 for cls, picking *just one* evidence
        oracle_truth, oracle_pred = zip(*[random.choice(x) for x in oracle_pipeline_predictions.values()])
        e2e_score(oracle_truth, oracle_pred, 'Oracle (one)', evidence_classes)

        # oracle classification F1 for cls, picking *all* evidence
        oracle_all_truth, oracle_all_pred = zip(*itertools.chain.from_iterable(oracle_pipeline_predictions.values()))
        e2e_score(oracle_all_truth, oracle_all_pred, 'Oracle (all)', evidence_classes)

        # for just the correctly predicted evidence spans, how does our final classification perform?
        # how hard is it when it's "easy" to find the evidence?
        correct_ev_truths, correct_ev_preds = zip(*correct_evidence_predictions.values())
        e2e_score(correct_ev_truths, correct_ev_preds, 'Correct evidence only', evidence_classes)

        # how well does our classifier do when we find an incorrect evidence?
        # for just the incorrectly predicted evidence spans, how does our final classification perform?
        # do we pick up on in document correlations? something else?
        # TODO: check if this is an evidence statement for a different ICO
        # TODO: check if this is an undiscovered evidence statement 
        # TODO: check how far down we need to go to find a known evidence statement for this ICO vs. some other ICO
        incorrect_ev_truths, incorrect_ev_preds = zip(*incorrect_evidence_predictions.values())
        e2e_score(incorrect_ev_truths, incorrect_ev_preds, 'Incorrect evidence only', evidence_classes)

        # for the incorrectly predicgted evidence spans, how would we have done if we used a correct span?
        # how hard is the classification when finding the evidence is hard?
        counterfactual_ev_truths, counterfactual_ev_preds = zip(*[random.choice(oracle_pipeline_predictions[key]) for key in set(incorrect_evidence_predictions.keys())])
        #counterfactual_ev_truths, counterfactual_ev_preds = zip(*[random.choice(oracle_pipeline_predictions[key]) for key in set(incorrect_evidence_predictions.keys()) & set(oracle_pipeline_predictions.keys())])
        e2e_score(counterfactual_ev_truths, counterfactual_ev_preds, 'Counterfactual evidence only', evidence_classes)

        # take the correct + the fixed ones
        # this should be similar to the Oracle score
        augmented_counterfactual_ev_truths = counterfactual_ev_truths + correct_ev_truths
        augmented_counterfactual_ev_preds = counterfactual_ev_preds + correct_ev_preds
        e2e_score(augmented_counterfactual_ev_truths, augmented_counterfactual_ev_preds, 'Augmented counterfactual evidence only', evidence_classes)

        # what happens if we use an unconditioned identifier
        if unconditioned_evidence_identifier is not None:
            logging.info('Unconditional identifier scores (note the classifier is conditional)')
            ev_only_unid_soft_pred = list(itertools.chain.from_iterable(all_unconditional_id_predictions[x] for x in annotations_with_evidence))
            id_scores(ev_only_id_truth, ev_only_unid_soft_pred, [top1_pipeline_unid_truth[x] for x in annotations_with_evidence], top1_cls_preds=top1_pipeline_cls_prediction, preds_dict=all_unconditional_id_predictions, preds_truth=all_id_truths)
            unid_cls_truth, unid_cls_pred = zip(*unconditioned_cls.values())
            e2e_score(unid_cls_truth, unid_cls_pred, "Unconditioned identifier (how important is the ICO to finding the evidence)", evidence_classes)
            # TODO how well does our conditioned evidence identifier recover *any* evidence (as determined by the unconditioned evidence identifier & all the data)

        def detok(sent):
            return ' '.join(detokenizer[x.item() if isinstance(x, torch.Tensor) else x] for x in sent)
        def view(id_predictions, unid_soft_predictions, id_truths, cls_predictions, cls_truths, ann, best, p=False):
            out = []
            out.append(f'ico: {detok(ann.i)} vs. {detok(ann.c)} for {detok(ann.o)}')
            if unid_soft_predictions is None:
                unid_soft_predictions = [torch.tensor([0.0, 0.0]) for _ in id_predictions]
            for idx, (idp, uidp, idt, clsp, clst, sent) in enumerate(zip(id_predictions, unid_soft_predictions, id_truths, cls_predictions, cls_truths, ann.tokenized_sentences)):
                best_marker = '*' if idx == best else ''
                out.append(detok(sent))
                out.append(f'  id pred{best_marker}: {idp[1].item():.3f}, unid id pred: {uidp[1].item():.3f}, id_truth: {idt}')
                out.append(f'  cls pred: {clsp}, cls truth: {clst}')
            if p:
                print('\n'.join(out))
            return '\n'.join(out)

        #import ipdb; ipdb.set_trace()
        total_length = 0
        debug_dir = os.path.join(save_dir, 'debug')
        incorrect_debug_dir = os.path.join(save_dir, 'debug_incorrect')
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(incorrect_debug_dir, exist_ok=True)
        for ann in data:
            key = (ann.prompt_id, ann.doc.docid)
            id_predictions = id_soft_pred[total_length:total_length + len(ann.tokenized_sentences)]
            hard_id_predictions = id_hard_pred[total_length:total_length + len(ann.tokenized_sentences)]
            #unid_hard_predictions = unid_hard_pred[total_legnth:total_length + len(ann.tokenized_sentences)]
            if unid_soft_pred is not None:
                unid_soft_predictions = unid_soft_pred[total_length:total_length + len(ann.tokenized_sentences)]
            else:
                unid_soft_predictions = None
            id_truths = id_truth[total_length:total_length + len(ann.tokenized_sentences)]
            cls_predictions = cls_hard_pred[total_length:total_length + len(ann.tokenized_sentences)]

            best_id_sent_idx = np.argmax([id_pred[1] for id_pred in id_predictions])
            id_tru = id_truth[total_length + best_id_sent_idx]
            cls_tru = cls_truth[total_length + best_id_sent_idx]
            cls_trus = cls_truth[total_length:total_length + len(ann.tokenized_sentences)]
            cls_pre = cls_hard_pred[total_length + best_id_sent_idx]

            pretty = view(id_predictions, unid_soft_predictions, id_truths, cls_predictions, cls_trus, ann, best_id_sent_idx, p=False)
            with open(os.path.join(debug_dir, str(ann.doc.docid) + '_' + str(ann.prompt_id) + '.txt'), 'w') as of:
                of.write(pretty)
            if id_tru == 0:
                with open(os.path.join(incorrect_debug_dir, str(ann.doc.docid) + '_' + str(ann.prompt_id) + '.txt'), 'w') as of:
                    of.write(pretty)
            total_length += len(ann.tokenized_sentences)


def train_module(model: nn.Module,
                 save_dir: str,
                 model_name: str,
                 train: List[Annotation],
                 val: List[Annotation],
                 #documents: Dict[str, List[List[int]]],
                 model_pars: dict,
                 sep_token_id: int,
                 sampler: Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]],
                 val_sampler: Callable[[Annotation], List[Tuple[torch.IntTensor, Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor], int]]],
                 optimizer=None,
                 scheduler=None,
                 detokenizer=None) -> Tuple[nn.Module, dict]:
    """Trains a module for evidence identification or classification.
    
    Loosely based on the work done for the ERASER Benchmark: DeYoung et al., 2019

    This method tracks loss on the entire validation set, saves intermediate
    models, and supports restoring from an unfinished state. The best model on
    the validation set is maintained, and the model stops training if a patience
    (see below) number of epochs with no improvement is exceeded.

    As there are likely too many negative examples to reasonably train a
    classifier on everything, every epoch we subsample the negatives.

    Args:
        model: some model like BertClassifier
        save_dir: a place to save intermediate and final results and models.
        model_name: a string for saving information
        train: a List of interned Annotation objects.
        val: a List of interned Annotation objects.
        #documents: a Dict of interned sentences
        model_pars: Arbitrary parameters directory, assumed to contain:
            lr: learning rate
            batch_size: an int
            sampling_method: a string, plus additional params in the dict to define creation of a sampler
            epochs: the number of epochs to train for
            patience: how long to wait for an improvement before giving up.
            max_grad_norm: optional, clip gradients.
        optimizer: what pytorch optimizer to use, if none, initialize Adam
        scheduler: optional, do we want a scheduler involved in learning?
        tensorize_model_inputs: should we convert our data to tensors before passing it to the model?
                                Useful if we have a model that performs its own tokenization (e.g. BERT as a Service)

    Returns:
        the trained evidence identifier and a dictionary of intermediate results.
    """

    logging.info(f'Beginning training {model_name} with {len(train)} annotations, {len(val)} for validation')
    output_dir = os.path.join(save_dir, f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model_save_file = os.path.join(output_dir, f'{model_name}.pt')
    epoch_save_file = os.path.join(output_dir, f'{model_name}_epoch_data.pt')

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr = model_pars['lr'])
    criterion = nn.CrossEntropyLoss(reduction='none')
    batch_size = model_pars['batch_size']
    if 'oracle' in model_name:
        batch_size = batch_size // 2
    epochs = model_pars['epochs']
    patience = model_pars['patience']
    max_grad_norm = model_pars.get('max_grad_norm', None)

    device = next(model.parameters()).device

    # TODO output an AUC where appropriate
    results = {
        # "sampled" losses do not represent the true data distribution, but do represent training data
        'sampled_epoch_train_acc': [],
        'sampled_epoch_train_losses': [],
        'sampled_epoch_train_f1': [],
        'sampled_epoch_val_acc': [],
        'sampled_epoch_val_losses': [],
        'sampled_epoch_val_f1': [],
        # "full" losses do represent the true data distribution
        'full_epoch_val_losses': [],
        'full_epoch_val_f1': [],
        'full_epoch_val_acc': [],
    }
    # allow restoring an existing training run
    start_epoch = 0
    best_epoch = -1
    best_val_loss = float('inf')
    best_val_f1 = float('-inf')
    best_model_state_dict = None
    epoch_data = {}
    if os.path.exists(epoch_save_file):
        model.load_state_dict(torch.load(model_save_file))
        epoch_data = torch.load(epoch_save_file)
        start_epoch = epoch_data['epoch'] + 1
        # handle finishing because patience was exceeded or we didn't get the best final epoch
        if bool(epoch_data.get('done', 0)):
            start_epoch = epochs
        results = epoch_data['results']
        best_epoch = start_epoch
        best_model_state_dict = OrderedDict({k:v.cpu() for k,v in model.state_dict().items()})
        logging.info(f'Restored training from epoch {start_epoch}')
    logging.info(f'Training evidence model from epoch {start_epoch} until epoch {epochs}')
    optimizer.zero_grad()
    sep = torch.tensor(sep_token_id, dtype=torch.int).unsqueeze(0)
    #import ipdb; ipdb.set_trace()
    for epoch in range(start_epoch, epochs):
        epoch_train_data = list(itertools.chain.from_iterable(sampler(t) for t in train))
        assert len(epoch_train_data) > 0
        train_classes = Counter(x[-1] for x in epoch_train_data)
        random.shuffle(epoch_train_data)
        epoch_val_data = list(itertools.chain.from_iterable(sampler(v) for v in val))
        assert len(epoch_val_data) > 0
        val_classes = Counter(x[-1] for x in epoch_val_data)
        random.shuffle(epoch_val_data)
        sampled_epoch_train_loss = 0
        model.train()
        logging.info(f'Training with {len(epoch_train_data) // batch_size} batches with {len(epoch_train_data)} examples')
        logging.info(f'Training classes distribution: {train_classes}, valing class distribution: {val_classes}')
        hard_train_preds = []
        hard_train_truths = []
        optimizer.zero_grad()
        for batch_start in range(0, len(epoch_train_data), batch_size):
            model.train()
            batch_elements = epoch_train_data[batch_start:min(batch_start+batch_size, len(epoch_train_data))]
            # we sample every time to thereoretically get a better representation of instances over the corpus.
            # this might just take more time than doing so in advance.
            sentences, queries, targets = zip(*filter(lambda x: x, batch_elements))
            hard_train_truths.extend(targets)
            #sentences = [s.to(device=device) for s in sentences]
            queries = [torch.cat([i, sep, c, sep, o]).to(dtype=torch.long) for (i,c,o) in queries]
            preds = model(queries, sentences)
            hard_train_preds.extend([x.cpu().item() for x in torch.argmax(preds, dim=-1)])
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            loss = criterion(preds, targets.to(device=preds.device)).sum()
            sampled_epoch_train_loss += loss.item()
            loss = loss / len(preds)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        sampled_epoch_train_loss /= len(epoch_train_data)
        results['sampled_epoch_train_losses'].append(sampled_epoch_train_loss)
        results['sampled_epoch_train_acc'].append(accuracy_score(hard_train_truths, hard_train_preds))
        results['sampled_epoch_train_f1'].append(classification_report(hard_train_truths, hard_train_preds, output_dict=True))
        logging.info(f'Epoch {epoch} sampled training loss {sampled_epoch_train_loss}, acc {results["sampled_epoch_train_acc"][-1]}')

        with torch.no_grad():
            model.eval()
            sampled_epoch_val_loss, _, sampled_epoch_val_hard_pred, sampled_epoch_val_truth = make_preds_epoch(model, epoch_val_data, batch_size, sep_token_id, device, criterion)
            sampled_epoch_val_acc = accuracy_score(sampled_epoch_val_truth, sampled_epoch_val_hard_pred)
            sampled_epoch_val_f1 = classification_report(sampled_epoch_val_truth, sampled_epoch_val_hard_pred, output_dict=True)
            results['sampled_epoch_val_losses'].append(sampled_epoch_val_loss)
            results['sampled_epoch_val_acc'].append(sampled_epoch_val_acc)
            results['sampled_epoch_val_f1'].append(sampled_epoch_val_f1)
            logging.info(f'Epoch {epoch} sampled val loss {sampled_epoch_val_loss}, acc {sampled_epoch_val_acc}, f1: {sampled_epoch_val_f1}')
            # evaluate over *all* of the validation data
            all_val_data = list(itertools.chain.from_iterable(val_sampler(v) for v in val))
            epoch_val_loss, epoch_val_soft_pred, epoch_val_hard_pred, epoch_val_truth = make_preds_epoch(model, all_val_data, batch_size, sep_token_id, device, criterion)
            results['full_epoch_val_losses'].append(epoch_val_loss)
            results['full_epoch_val_acc'].append(accuracy_score(epoch_val_truth, epoch_val_hard_pred))
            results['full_epoch_val_f1'].append(classification_report(epoch_val_truth, epoch_val_hard_pred, output_dict=True))
            logging.info(f'Epoch {epoch} full val loss {epoch_val_loss}, accuracy: {results["full_epoch_val_acc"][-1]}, f1: {results["full_epoch_val_f1"][-1]}')

            full_val_f1 = results['full_epoch_val_f1'][-1]['macro avg']['f1-score']
            #if epoch_val_loss < best_val_loss:
            #if sampled_epoch_val_loss < best_val_loss:
            if full_val_f1 > best_val_f1:
                logging.debug(f'Epoch {epoch} new best model with full val f1 {full_val_f1}')
                #logging.debug(f'Epoch {epoch} new best model with sampled val loss {sampled_epoch_val_loss}')
                best_model_state_dict = OrderedDict({k:v.cpu() for k,v in model.state_dict().items()})
                best_epoch = epoch
                best_val_f1 = full_val_f1
                #best_val_loss = sampled_epoch_val_loss
                torch.save(model.state_dict(), model_save_file)
                epoch_data = {
                    'epoch': epoch,
                    'results': results,
                    'best_val_loss': best_val_loss,
                    'done': 0
                }
                torch.save(epoch_data, epoch_save_file)
        if epoch - best_epoch > patience:
            epoch_data['done'] = 1
            torch.save(epoch_data, epoch_save_file)
            break

    epoch_data['done'] = 1
    epoch_data['results'] = results
    torch.save(epoch_data, epoch_save_file)
    model.load_state_dict(best_model_state_dict)
    model = model.to(device=device)
    model.eval()
    return model, results

def load_data(save_dir: str, params: dict, tokenizer, evidence_classes: Dict[str, int]):
    data_file = os.path.join(save_dir, 'datasets.pkl')
    articles_file = os.path.join(save_dir, 'bert_articles.pkl')
    use_abstracts = bool(params.get('use_abstracts', False))
    if os.path.exists(data_file):
        logging.info(f'Resurrecting train/val/test from {data_file}')
        # this is a hack to deal with a previous version of data loading
        train, val, test = torch.load(data_file)
        if len(test) == 0:
            logging.info(f'Test was empty, using val')
            test = val
        return (train, val, test)
    ei_annotations = preprocessor.read_annotations()
    ei_prompts = preprocessor.read_prompts()
    if use_abstracts:
        ei_annotations = ei_annotations[ei_annotations['In Abstract']]
    # TODO these column names should be factored out
    logging.info('joining prompts/annotations')
    joined = ei_annotations.merge(ei_prompts, on=PROMPT_ID_COL_NAME, suffixes=('', '_y'))

    article_ids = set(joined[STUDY_ID_COL].unique())
    if os.path.exists(articles_file):
        logging.info(f'Reading {len(article_ids)} computed articles from {articles_file}')
        bert_articles = torch.load(articles_file)
    else:
        logging.info(f'Reading {len(article_ids)} articles')
        articles = preprocessor.read_in_text_articles(article_ids, abstracts=use_abstracts)
        #tokenizer = BertTokenizer.from_pretrained(params['bert_vocab'])
        logging.info(f'Converting {len(articles)} articles for BERT')
        bert_articles = dict(map(lambda x: (x.get_pmcid(), retokenize_with_bert(to_structured(x), tokenizer)), articles))
        torch.save(bert_articles, articles_file)

    train_file = params['train_data']
    val_file = params['val_data']
    test_file = params['test_data']
    train_ids = preprocessor._read_ids(train_file)
    val_ids = preprocessor._read_ids(val_file)
    test_ids = preprocessor._read_ids(test_file)
    all_ids = train_ids | val_ids | test_ids
    assert len(train_ids & test_ids) == 0
    assert len(train_ids & val_ids) == 0
    target_ids = set(train_ids | val_ids | test_ids)
    train, val, test, skipped = [], [], [], []

    logging.info('Converting prompts')
    prompt_ids = set(joined[PROMPT_ID_COL_NAME].unique())
    skipped_prompts = set()

    for prompt_id in prompt_ids:
        anns = joined[joined[PROMPT_ID_COL_NAME] == prompt_id]
        (docid,) = anns[STUDY_ID_COL].unique()
        docid = str(docid)
        if docid not in bert_articles:
            logging.warn(f'Skipping prompt {prompt_id} for missing document {docid}')
            skipped_prompts.add(prompt_id)
            continue
        (i,) = anns['Intervention'].unique()
        (c,) = anns['Comparator'].unique()
        (o,) = anns['Outcome'].unique()
        spans = anns[[EVIDENCE_START, EVIDENCE_END]]
        label = evidence_classes[stats.mode(anns[LABEL])[0][0]]
        doc = bert_articles[docid]
        sentences = list(doc.sentences)
        evidence_texts = anns[EVIDENCE_COL_NAME]
        ev_spans = []
        for (_, row) in spans.iterrows():
            start = row[EVIDENCE_START]
            end = row[EVIDENCE_END]
            ev_spans.append((start, end))
            ev_chars = set(range(start, end))
            if start == -1: # skip unrecovered evidence spans
                continue
            span = doc.sentence_span(start, end)
            if span is None:
                continue
            for si in range(*span):
                char_range = set(range(sentences[si].start_offset, sentences[si].end_offset))
                intersection = char_range & ev_chars
                # if the size of the intersction with our evidence is:
                # (1) very small AND
                # (2) it does not entirely contain the evidence
                # then we skip this sentence.
                if len(intersection) < .2 * len(char_range) and len(intersection) < .9 * len(ev_chars):
                    continue
                sent = asdict(sentences[si])
                sent['labels'] = sent['labels'] if sent['labels'] else dict()
                sent['labels']['evidence'] = 1
                tokens = sent['tokens']
                for t in tokens:
                    if t.get('labels', None) is None:
                        t['labels'] = dict()
                    if len(set(range(t['start_offset'], t['end_offset'])) & ev_chars) > 0:
                        t['labels']['evidence'] = 1
                    elif 'evidence' not in t:
                        t['labels']['evidence'] = 0
                sent['tokens'] = tuple(tokens)
                sentences[si] = from_dict(data_class=Sentence, data=sent)
        doc_parts = asdict(doc)
        doc_parts['sentences'] = tuple(sentences)
        doc = from_dict(data_class=Document, data=doc_parts)
        evidence_vector = torch.LongTensor([t.labels['evidence'] if t.labels is not None else 0 for t in doc.tokens()])
        ann = Annotation(doc=doc,
                         prompt_id=str(prompt_id),
                         tokenized_sentences=[torch.IntTensor([t.token_id for t in s.tokens]) for s in doc.sentences],
                         i=i,
                         c=c,
                         o=o,
                         evidence_texts=tuple(set(evidence_texts.values)),
                         evidence_spans=tuple(set(ev_spans)),
                         evidence_vector=evidence_vector,
                         significance_class=label).retokenize(tokenizer)

        docid = int(docid)
        if docid in train_ids:
            train.append(ann)
        if docid in val_ids:
            val.append(ann)
        if docid in test_ids:
            test.append(ann)
        if docid not in all_ids:
            skipped.append(docid)
    logging.info(f'Skipped {len(skipped_prompts)} prompts for missing documents')
    logging.info(f'Skipped converting {len(skipped)} ids')
    logging.info(f'Have {len(train)} training instances, {len(val)} validation instances, {len(test)} test instances')
    torch.save([train, val, test], data_file)
    #import ipdb; ipdb.set_trace()
    return train, val, test

def main():
    parser = argparse.ArgumentParser(description="""Trains a pipeline model.

    Loosely based on the pipeline in the ERASER Benchmark, DeYoung et al., 2019
    Step 1 is evidence identification, that is identify if a given sentence is evidence or not
    Step 2 is evidence classification, that is given an evidence sentence, classify the final outcome for the final task (e.g. sentiment or significance).

    These models should be separated into two separate steps, but at the moment:
    * prep data 
    * convert data for evidence identification - in the case of training data we take all the positives and sample some negatives
        * side note: this sampling is *somewhat* configurable and is done on a per-batch/epoch basis in order to gain a broader sampling of negative values.
    * train evidence identification
    * convert data for evidence classification - take all rationales + decisions and use this as input
    * train evidence classification
    * decode first the evidence, then run classification for each split
    
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--output_dir', dest='output_dir', required=True, help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--params', dest='params', required=True, help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    parser.add_argument('--data_only', dest='data_only', required=False, action='store_true', help='Process data only, no training')
    args = parser.parse_args()
    with open(args.params, 'r') as fp:
        logger.info(f'Loading model parameters from {args.params}')
        params = json.load(fp)
        logger.info(f'Params: {json.dumps(params, indent=2, sort_keys=True)}')
    evidence_identifier, evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer = initialize_models(params, '[UNK]')
    oracle_evidence_classifier = copy.deepcopy(evidence_classifier)
    ico_only_evidence_classifier = copy.deepcopy(evidence_classifier)
    unconditioned_oracle_evidence_classifier = copy.deepcopy(evidence_classifier)
    unconditioned_evidence_identifier = copy.deepcopy(evidence_identifier)
    unconditioned_evidence_classifier = copy.deepcopy(evidence_classifier)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copyfile(args.params, os.path.join(args.output_dir, os.path.basename(args.params)))
    train, val, test = load_data(args.output_dir, params, tokenizer, evidence_classes)
    assert len(test) > 0
    #train = train[:10]
    #val = train
    #test = train
    logging.info(f'Loaded {len(train)} training instances, {len(val)} valing instances, and {len(test)} testing instances')
    if args.data_only:
        sys.exit(0)
    evidence_classifier, evidence_classifier_training_results = train_module(evidence_classifier.cuda(),
                                                                             args.output_dir,
                                                                             "evidence_classifier",
                                                                             train,
                                                                             val,
                                                                             params['evidence_classifier'],
                                                                             tokenizer.sep_token_id,
                                                                             get_classifier_sampler(params),
                                                                             get_classifier_oracle_sampler(params),
                                                                             detokenizer=de_interner)
    evidence_classifier = evidence_classifier.cpu()
    evidence_identifier, evidence_identifier_training_results = train_module(evidence_identifier.cuda(),
                                                                             args.output_dir,
                                                                             "evidence_identifier",
                                                                             train,
                                                                             val,
                                                                             params['evidence_identifier'],
                                                                             tokenizer.sep_token_id,
                                                                             get_identifier_sampler(params),
                                                                             identifier_everything_sampler,
                                                                             detokenizer=de_interner)
    evidence_identifier = evidence_identifier.cpu()
    if unconditioned_evidence_identifier is not None:
        unconditioned_evidence_identifier, unconditioned_evidence_identifier_training_results = train_module(unconditioned_evidence_identifier.cuda(),
                                                                                                             args.output_dir,
                                                                                                             "unconditioned_evidence_identifier",
                                                                                                             train,
                                                                                                             val,
                                                                                                             params['evidence_identifier'],
                                                                                                             tokenizer.sep_token_id,
                                                                                                             mask_sampler(get_identifier_sampler(params), {'i', 'c', 'o'}),
                                                                                                             mask_sampler(identifier_everything_sampler, {'i', 'c', 'o'}),
                                                                                                             detokenizer=de_interner)
        unconditioned_evidence_identifier = unconditioned_evidence_identifier.cpu()
        unconditioned_evidence_classifier, unconditioned_evidence_classifier_training_results = train_module(unconditioned_evidence_classifier.cuda(),
                                                                                                             args.output_dir,
                                                                                                             "unconditioned_evidence_classifier",
                                                                                                             train,
                                                                                                             val,
                                                                                                             params['evidence_classifier'],
                                                                                                             tokenizer.sep_token_id,
                                                                                                             mask_sampler(get_classifier_sampler(params), {'i', 'c', 'o'}),
                                                                                                             mask_sampler(get_classifier_oracle_sampler(params), {'i', 'c', 'o'}),
                                                                                                             detokenizer=de_interner)
        unconditioned_evidence_classifier = unconditioned_evidence_classifier.cpu()
    for t, d in [('val', val), ('test', test)]:
        logging.info(f"\n\n\n\nConditioned scores {t}\n\n\n\n")
        decode(evidence_identifier.cuda(),
               evidence_classifier.cuda(),
               None,
               #unconditioned_evidence_identifier.cuda() if unconditioned_evidence_identifier else None,
               args.output_dir,
               t,
               d,
               params,
               tokenizer.sep_token_id,
               identifier_everything_sampler,
               classifier_everything_sampler,
               detokenizer=de_interner)
        logging.info(f"\n\n\n\nUnconditioned scores {t}\n\n\n\n")
        decode(unconditioned_evidence_identifier.cuda(),
               unconditioned_evidence_classifier.cuda(),
               None,
               args.output_dir,
               f"{t}_unconditioned",
               d,
               params,
               tokenizer.sep_token_id,
               identifier_everything_sampler,
               classifier_everything_sampler,
               detokenizer=de_interner)
    # TODO fix the data for this so we union evidence sentences across document
    #evidence_identifier = evidence_identifier.cpu()
    #evidence_classifier = evidence_classifier.cpu()

    if oracle_evidence_classifier is not None:
        logging.info('Conditioned oracle evidence classifier')
        oracle_sampler = get_classifier_oracle_sampler(params)
        oracle_evidence_classifier, oracle_evidence_classifier_training_results = train_module(oracle_evidence_classifier.cuda(),
                                                                                               args.output_dir,
                                                                                               "oracle_evidence_classifier",
                                                                                               train,
                                                                                               val,
                                                                                               params['evidence_classifier'],
                                                                                               tokenizer.sep_token_id,
                                                                                               oracle_sampler,
                                                                                               oracle_sampler,
                                                                                               detokenizer=de_interner)
        for n, t in [('val', val), ('test', test)]:
            logging.info(f'Decoding on {n}')
            _, _, oracle_hard_pred, oracle_tru = make_preds_epoch(oracle_evidence_classifier.cuda(),
                                                                  [instance.to_model_input('classifier') for instance in oracle_decoding_instances(t)],
                                                                  tokenizer.sep_token_id, params['evidence_classifier']['batch_size'],
                                                                  device='cuda:0',
                                                                  criterion=None)
            e2e_score(oracle_tru, oracle_hard_pred, 'Conditioned oracle scoring', evidence_classes)
        oracle_evidence_classifier = oracle_evidence_classifier.cpu()
    if unconditioned_oracle_evidence_classifier is not None:
        logging.info('Unconditioned oracle evidence classifier')
        oracle_sampler = get_classifier_oracle_sampler(params)
        unconditioned_oracle_evidence_classifier, unconditioned_oracle_evidence_classifier_training_results = train_module(unconditioned_oracle_evidence_classifier.cuda(),
                                                                                                                           args.output_dir,
                                                                                                                           "unconditioned_oracle_evidence_classifier",
                                                                                                                           train,
                                                                                                                           val,
                                                                                                                           params['evidence_classifier'],
                                                                                                                           tokenizer.sep_token_id,
                                                                                                                           mask_sampler(oracle_sampler, {'i', 'c', 'o'}),
                                                                                                                           mask_sampler(oracle_sampler, {'i', 'c', 'o'}),
                                                                                                                           detokenizer=de_interner)
        for n, t in [('val', val), ('test', test)]:
            logging.info(f'Decoding on {n}')
            _, _, oracle_hard_pred, oracle_tru = make_preds_epoch(unconditioned_oracle_evidence_classifier.cuda(),
                                                                  [instance.to_model_input('unconditioned_classifier') for instance in oracle_decoding_instances(t)],
                                                                  tokenizer.sep_token_id, params['evidence_classifier']['batch_size'],
                                                                  device='cuda:0',
                                                                  criterion=None)
            e2e_score(oracle_tru, oracle_hard_pred, 'Unconditioned oracle scoring', evidence_classes)
        unconditioned_oracle_evidence_classifier = unconditioned_oracle_evidence_classifier.cpu()
    if ico_only_evidence_classifier is not None:
        logging.info('ICO only evidence classifier')
        oracle_sampler = get_classifier_oracle_sampler(params)
        ico_only_evidence_classifier, ico_only_evidence_classifier_training_results = train_module(ico_only_evidence_classifier.cuda(),
                                                                                                   args.output_dir,
                                                                                                   "ico_only_evidence_classifier",
                                                                                                   train,
                                                                                                   val,
                                                                                                   params['evidence_classifier'],
                                                                                                   tokenizer.sep_token_id,
                                                                                                   mask_tokens(oracle_sampler, tokenizer.pad_token_id),
                                                                                                   mask_tokens(oracle_sampler, tokenizer.pad_token_id),
                                                                                                   detokenizer=de_interner)
        for n, t in [('val', val), ('test', test)]:
            logging.info(f'Decoding on {n}')
            _, _, ico_only_hard_pred, ico_only_tru = make_preds_epoch(ico_only_evidence_classifier.cuda(),
                                                                      [instance.to_model_input('ico_only') for instance in oracle_decoding_instances(t)],
                                                                      tokenizer.sep_token_id, params['evidence_classifier']['batch_size'],
                                                                      device='cuda:0',
                                                                      criterion=None)
            e2e_score(ico_only_tru, ico_only_hard_pred, 'ICO only scoring', evidence_classes)
        ico_only_evidence_classifier = ico_only_evidence_classifier.cpu()

if __name__ == '__main__':
    main()
