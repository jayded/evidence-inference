from collections import OrderedDict
import os 

import pandas as pd 
import numpy as np
np.random.seed(896)

from os.path import join, dirname, abspath
import sys

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

from sklearn.feature_extraction.text import CountVectorizer
from evidence_inference.preprocess.sentence_split import find_span_location, split_into_sentences, gen_exact_evid_array
from evidence_inference.preprocess.article_reader import TableHTMLParser
import evidence_inference.preprocess.article_reader as article_reader
parser = TableHTMLParser()

# this monstrosity points us to a root directory relative to this file
annotation_root = os.path.abspath(os.path.join(__file__, "..", "..", "..", "annotations"))
anno_csv_path = os.path.join(annotation_root, "annotations_merged.csv")  # "pilot_run_data/annotations.csv"
prompts_csv_path = os.path.join(annotation_root, "prompts_merged.csv")  # "pilot_run_data/prompts.csv"
base_XML_path = os.path.join(annotation_root, "xml_files")
_train_id_file, _validation_id_file, _test_id_file = [os.path.join(annotation_root, 'splits', d) for d in ['train_article_ids.txt', 'validation_article_ids.txt', 'test_article_ids.txt']]

if not all(os.path.exists(x) for x in [anno_csv_path, prompts_csv_path, base_XML_path]):
    raise RuntimeError("One of {} does not exist".format([anno_csv_path, prompts_csv_path, base_XML_path]))

# base_XML_path = os.path.join("pilot_run_data", "xml_files")

PROMPT_ID_COL_NAME = "PromptID"
LBL_COL_NAME = "Label Code"
LABEL = 'Label'
EVIDENCE_COL_NAME = "Annotations"
EVIDENCE_START = "Evidence Start"
EVIDENCE_END = "Evidence End"
STUDY_ID_COL = "PMCID"
VALID_LABEL = "Valid Label"
VALID_REASONING = "Valid Reasoning"

def get_article(article_id):
    xml_str = "PMC{}.nxml".format(article_id)
    xml_path = os.path.join(base_XML_path, xml_str)
    return article_reader.Article(xml_path)

def read_in_articles(article_ids=None):

    anno_df = pd.read_csv(anno_csv_path)
    unique_article_ids = anno_df[STUDY_ID_COL].unique()

    articles = []

    for article_id in unique_article_ids:
        #import pdb; pdb.set_trace()

        if article_ids is None or article_id in article_ids:
            # 2376383
            articles.append(get_article(article_id))

    return articles


def extract_raw_text(article, sections_of_interest=None):
    if sections_of_interest is None:
        #sections_of_interest = ["results", ""]
        sections_of_interest = article.article_dict.keys()
   
    ti_ab = "TITLE: " + article.get_title() + "\n\n"

    article_sections = [sec for sec in article.article_dict.keys() if any(
                            [s in sec for s in sections_of_interest])]
    article_body = article.to_raw_str(fields=article_sections)
    
    raw_text = ti_ab + "  " + article_body
        
    return raw_text.replace("<p>", "")


def extract_text_from_prompts(prompts_df):
    I, C, O = prompts_df['Intervention'].values, prompts_df['Comparator'].values, prompts_df['Outcome'].values
    all_prompt_text = [s.lower() for s in np.concatenate([I, C, O])]
    return all_prompt_text


def get_inference_vectorizer(article_ids=None, sections_of_interest=None, vocabulary_file=None):

    # if article_ids is None, will use all articles
    # in the CSV passed to the read_in_articles method.
    articles = read_in_articles(article_ids=article_ids)
    raw_texts = [extract_raw_text(article, sections_of_interest) for article in articles]

    # we also use the prompts text to construct our vectorizer
    prompts = read_prompts()
    raw_prompt_text = " ".join(extract_text_from_prompts(prompts))

    raw_texts.append(raw_prompt_text)

    # there is at least one prompt with tokens short enough that CountVectorizer's default destroys it, so we allow any single character through.
    if vocabulary_file is not None:
        with open(vocabulary_file, 'r') as vf:
            vocab = [line.strip() for line in vf]
        vectorizer = CountVectorizer(vocabulary=vocab, token_pattern=r"\b\w+\b")
        print("Loaded {} words from vocab file {}".format(len(vocab), vocabulary_file))
    else:
        vectorizer = CountVectorizer(max_features=20000, token_pattern=r"\b\w+\b")
    vectorizer.fit(raw_texts)
    tokenizer = vectorizer.build_tokenizer() 

    str_to_idx = vectorizer.vocabulary_
    str_to_idx[SimpleInferenceVectorizer.PAD] = max(vectorizer.vocabulary_.values())
    str_to_idx[SimpleInferenceVectorizer.UNK] = str_to_idx[SimpleInferenceVectorizer.PAD]+1
    
    # note that for now the vectorizer is fit using only the
    # article texts (i.e., the vocab is based on words in full-texts,
    # not in prompts necessarily).
    return SimpleInferenceVectorizer(str_to_idx, tokenizer)


def read_annotations():
    anno_df = pd.read_csv(anno_csv_path)
    # we need to force EVIDENCE_COL_NAME to be strings in all cases; pandas occasionally reads some values as floats.
    anno_df = anno_df[anno_df.apply(lambda row: bool(row[VALID_LABEL]) and bool(row[VALID_REASONING]) and len(str(row[EVIDENCE_COL_NAME])) > 0 and row[LABEL] != 'invalid prompt', axis=1)]
    #annos[~annos["Answer_Val"].isin([-1, 0, 1])]
    # TODO revisit this; right now just overwriting for convienence
    # anno_df["Answer_Val"].replace({3:0}, inplace=True)
    return anno_df


def read_prompts():
    prompts_df = pd.read_csv(prompts_csv_path)
    prompts_df = prompts_df[prompts_df.apply(lambda row: all(map(lambda x: type(x) == str and x is not None and bool(x.strip()), [row['Comparator'], row['Intervention'], row['Outcome']])), axis=1)]
    return prompts_df 


def assemble_Xy_for_prompts(training_prompts, inference_vectorizer, lbls_too=False, annotations=None, sections_of_interest=None, include_sentence_span_splits = False): 
    Xy = []
    for prompt_id in training_prompts[PROMPT_ID_COL_NAME].values:
        if lbls_too:
            Xy_dict = inference_vectorizer.vectorize(training_prompts, prompt_id, 
                                include_lbls=True, annotations_df=annotations, sections_of_interest=sections_of_interest, include_sentence_span_splits = include_sentence_span_splits)
        else:
            Xy_dict = inference_vectorizer.vectorize(training_prompts, prompt_id, sections_of_interest=sections_of_interest, include_sentence_span_splits = include_sentence_span_splits)
        Xy.append(Xy_dict)
    return Xy


def train_document_ids():
    """ Returns the set of document ids for a fixed training set """
    with open(_train_id_file, 'r') as tf:
        ids = list(int(x.strip()) for x in tf.readlines())
        ids_dict = OrderedDict()
        for x in ids:
            ids_dict[x] = x
        return ids_dict.keys()

def validation_document_ids():
    """ Returns the set of document ids for a fixed validation set """
    with open(_validation_id_file, 'r') as vf:
        ids = list(int(x.strip()) for x in vf.readlines())
        ids_dict = OrderedDict()
        for x in ids:
            ids_dict[x] = x
        return ids_dict.keys()

def test_document_ids():
    """ Returns the set of documents for a fixed test set """
    with open(_test_id_file, 'r') as tf:
        ids = list(int(x.strip()) for x in tf.readlines())
        ids_dict = OrderedDict()
        for x in ids:
            ids_dict[x] = x
        return ids_dict.keys()

def get_train_Xy(train_doc_ids, sections_of_interest=None, vocabulary_file=None, include_sentence_span_splits = False):
    """ Loads the relevant documents, builds a vectorizer, and returns a list of training instances"""
    prompts = read_prompts()
    annotations = read_annotations()

    # filter out prompts for which we do not have annotations for whatever reason
    # this was actually just one case; not sure what was going on there.
    def have_annotations_for_prompt(prompt_id):
        return len(annotations[annotations[PROMPT_ID_COL_NAME] == prompt_id]) > 0

    prompts = [prompt for row_idx, prompt in prompts.iterrows() if 
                            have_annotations_for_prompt(prompt[PROMPT_ID_COL_NAME])]
    prompts = pd.DataFrame(prompts)

    inference_vectorizer = get_inference_vectorizer(article_ids=train_doc_ids, sections_of_interest=sections_of_interest, vocabulary_file=vocabulary_file)

    training_prompts = prompts[prompts[STUDY_ID_COL].isin(train_doc_ids)]

    training_prompts = pd.DataFrame(training_prompts)
    train_Xy = assemble_Xy_for_prompts(training_prompts, inference_vectorizer, lbls_too=True, annotations=annotations, include_sentence_span_splits = include_sentence_span_splits)

    return train_Xy, inference_vectorizer


def get_Xy(docids, inference_vectorizer: 'SimpleInferenceVectorizer', sections_of_interest=None, include_sentence_span_splits = False):
    prompts = read_prompts()
    annotations = read_annotations()

    # filter out prompts for which we do not have annotations for whatever reason
    # this was actually just one case; not sure what was going on there.
    def have_annotations_for_prompt(prompt_id):
        return len(annotations[annotations[PROMPT_ID_COL_NAME] == prompt_id]) > 0

    prompts = [prompt for row_idx, prompt in prompts.iterrows() if
               have_annotations_for_prompt(prompt[PROMPT_ID_COL_NAME])]
    prompts = pd.DataFrame(prompts)

    prompts = prompts[prompts[STUDY_ID_COL].isin(docids)]
    Xy = assemble_Xy_for_prompts(prompts, inference_vectorizer, lbls_too=True, annotations=annotations, sections_of_interest=sections_of_interest, include_sentence_span_splits = include_sentence_span_splits)
    return Xy


class SimpleInferenceVectorizer:
    UNK = "<unk>"
    PAD = "<pad>"

    def __init__(self, str_to_idx, tokenizer):
        self.str_to_idx = str_to_idx
        self.idx_to_str = [None]*(len(self.str_to_idx))
        self.sentence_splits = {} # map of article ids to array of sentence splits
        self.token_evidence  = {}
        
        for w, idx in self.str_to_idx.items():
            try:
                self.idx_to_str[idx] = w 
            except:
                import pdb; pdb.set_trace()

        self.tokenizer = tokenizer

    def string_to_seq(self, s):
        tokenized = self.tokenizer(s)
        unk_idx = self.str_to_idx[SimpleInferenceVectorizer.UNK]
        vectorized = [self.str_to_idx.get(token, unk_idx) for token in tokenized]
        return np.array(vectorized)

    def vectorize(self, prompts_df, prompt_id, include_lbls=False, annotations_df=None, sections_of_interest=None, include_sentence_span_splits = False):
        """
        Vectorize the prompt specified by the ID.
        """
        if include_lbls and annotations_df is None:
            raise ValueError("When including annotations, they must already be defined")

        prompt = prompts_df[prompts_df[PROMPT_ID_COL_NAME]==prompt_id]

        ###
        # vectorize the article itself.
        article_id = str(prompt[STUDY_ID_COL].values[0])
        article = get_article(article_id)
        article_text = extract_raw_text(article, sections_of_interest)
        article_text = article_text.lower()
        vectorized_article = self.string_to_seq(article_text)
        
        ###
        # and now vectorize the prompt (I/C/O)
        I_v = self.string_to_seq(prompt["Intervention"].values[0].lower())
        C_v = self.string_to_seq(prompt["Comparator"].values[0].lower())
        O_v = self.string_to_seq(prompt["Outcome"].values[0].lower())

        return_dict = {"article":vectorized_article, "I":I_v, "C":C_v, "O":O_v, "a_id": article_id, "p_id": prompt_id}

        if include_lbls:
            # then also read out the labels.
            assert (annotations_df is not None)
            annotations_for_prompt = annotations_df[annotations_df[PROMPT_ID_COL_NAME] == prompt_id]
            labels = annotations_for_prompt[[LBL_COL_NAME,EVIDENCE_COL_NAME]].values
            return_dict["y"] = labels
            # remove html tags
            for l in labels:
                parser.feed(str(l[1]))
                l[1] = parser.get_data()
            
            spans = annotations_for_prompt[[EVIDENCE_START,EVIDENCE_END]].values
            if len(spans) > 0 and sections_of_interest is None:
                # split into sentences, find which are evidence, and also encode all.
                sentence_spans = []
                if include_sentence_span_splits:
                    sen = split_into_sentences(article_id, article_text, self.sentence_splits)
                    tmp = find_span_location(sen, [s[0] for s in spans], [e[1] for e in spans])
                    for t in tmp:
                        sentence_spans.append([self.string_to_seq(t[0]), t[1]])
                 
                # encode the evidence spans 
                evidence_spans = set()
                for start, end in spans:
                    article_before_span = article_text[:int(start)]
                    # +1 because the slice gets every character (and therefore token) *before* the evidence, so we want to offset the token count by 1 to actually start in the evidence
                    span_start_idx = len(self.tokenizer(article_before_span)) + 1
                    article_at_end_of_span = article_text[:int(end)]
                    span_end_idx = len(self.tokenizer(article_at_end_of_span))
                    evidence_spans.add((span_start_idx, span_end_idx))
                   
                return_dict['sentence_span'] = sentence_spans  
                return_dict['evidence_spans'] = evidence_spans
                if include_sentence_span_splits:
                    return_dict['token_ev_labels'] = gen_exact_evid_array(sentence_spans, evidence_spans, return_dict, self.idx_to_str)
                
        return return_dict

    def decode(self, v):
        return [self.idx_to_str[idx] for idx in v]