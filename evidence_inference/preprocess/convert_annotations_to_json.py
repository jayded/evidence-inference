import argparse
from collections import namedtuple
import os 
import json
import random

from os.path import join, dirname, abspath
import sys

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

from evidence_inference.preprocess import preprocessor
from preprocessor import read_prompts, read_annotations, train_document_ids, validation_document_ids, test_document_ids, PROMPT_ID_COL_NAME, STUDY_ID_COL, LABEL, EVIDENCE_COL_NAME, EVIDENCE_START, EVIDENCE_END

Annotation = namedtuple('Annotation', 'prompt_id article_id intervention comparator outcome label evidence_text evidence_start evidence_end')

def annotation_contains(a1: Annotation, a2: Annotation):
    if a1.evidence_start <= a2.evidence_start \
        and a2.evidence_start <= a1.evidence_end \
        and a1.evidence_end >= a2.evidence_end:
        return True
    return False

def extract_annotations(article_ids, output, collapse_evidence):
    all_annotations = dict()
    prompts_df = read_prompts()
    prompts_df = prompts_df[prompts_df[STUDY_ID_COL].isin(article_ids)]
    annotations_df = read_annotations()
    annotations_df = annotations_df[annotations_df[STUDY_ID_COL].isin(article_ids)]
    for _, (pid, aid, i, c, o) in prompts_df[[PROMPT_ID_COL_NAME, STUDY_ID_COL, 'Intervention', 'Comparator', 'Outcome']].iterrows():
        annotations = []
        filtered_annotations = annotations_df[annotations_df[PROMPT_ID_COL_NAME] == pid][[LABEL, EVIDENCE_COL_NAME, EVIDENCE_START, EVIDENCE_END]]
        for _, (label, evidence_text, evidence_start, evidence_end) in filtered_annotations.iterrows():
            if evidence_text != evidence_text:
                evidence_text = None
            annotations.append(Annotation(pid, aid, i, c, o, label, evidence_text, evidence_start, evidence_end))
        # sort annotations by first the start index, then the end
        annotations.sort(key=lambda x: x.evidence_end)
        annotations.sort(key=lambda x: x.evidence_start)
        if collapse_evidence:
            # find unique annotations (those not entirely contained within another)
            i = 0
            while i < len(annotations):
                j = i + 1
                while j < len(annotations):
                    # the second condition here is so we don't delete any (potentially) disagreeing annotations.
                    if annotation_contains(annotations[j], annotations[i])\
                        and annotations[i].label == annotations[j].label:
                        del annotations[i]
                        i -= 1
                        break
                    j += 1
                i += 1
            del i
        annotations = [a._asdict() for a in annotations]
        all_annotations[pid] = annotations
    with open(output, 'w') as of:
        json.dump(all_annotations, of, indent=2)

def get_document_ids(mode):
    if mode == 'experiment':
        random.seed(177)
        real_train_docs = list(preprocessor.train_document_ids())
        random.shuffle(real_train_docs)
        split_index = int(len(real_train_docs) * .9)
        train_docs = real_train_docs[:split_index]
        val_docs = real_train_docs[split_index:]
        return train_docs, val_docs, test_document_ids()
    elif mode == 'paper':
        return train_document_ids(), validation_document_ids(), test_document_ids()
    else:
        raise ValueError('implement me!')

def main():
    parser = argparse.ArgumentParser(description="Convert annotations to json.")
    parser.add_argument('--mode', dest='mode', required=True, type=str, help='paper|experiment : Should our "eval" set be the real test set, or should it be the validation set. In the latter case, a fake validation set from the training data will be generated. The "experiment" flag is most useful for those who monitor performance on the validation set or use both the training and validation sets for training.')
    parser.add_argument('--output_dir', dest='output_dir', required=True, type=str, help='Where do we want to write our json files?')
    parser.add_argument('--collapse_evidence', dest='collapse_evidence', action='store_true', default=False, help='Should we attempt to collapse mostly evidence rationales that: (1) are entirely contained within another rationale, and (2) have the same significance label as the containing rationale?')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_ids, val_ids, test_ids = get_document_ids(args.mode)
    extract_annotations(train_ids, os.path.join(args.output_dir, 'train.annotations.json'), args.collapse_evidence)
    extract_annotations(val_ids, os.path.join(args.output_dir, 'val.annotations.json'), args.collapse_evidence)
    extract_annotations(test_ids, os.path.join(args.output_dir, 'test.annotations.json'), args.collapse_evidence)

if __name__ == '__main__':
    main()
