import os
import sys

from dataclasses import dataclass, InitVar
from itertools import chain
from os.path import join, dirname, abspath
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import spacy
nlp = spacy.load('en_core_sci_sm')

from frozendict import frozendict

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))
from evidence_inference.preprocess.article_reader import Article, TextArticle

# TODO figure out how to make immutability and __post_init__ var setting work together
@dataclass(frozen=False, repr=True, eq=True)
class Token:
    raw_text: str # original text, kind-of-sort-of unprocessed. assumed to match slicing the source document with the start/end offsets
    start_offset: int #inclusive, character offset into source text
    end_offset: int #exclusive, characer offset into source text
    text: Optional[str] # arbitrarily processed version of raw_text. raw_text will be used if this is not defined
    token_id: Optional[int] # some kind of embedding id
    labels: Optional[Dict[str, Any]] # arbitrary collection of labels, user defined and optional

    #def __post_init__(self, text, token_id, labels):
    #    if text is None:
    #        self.text = self.raw_text
    #    assert self.raw_text is not None
    #    if labels is None:
    #        self.labels = None
    #    else:
    #        self.labels = frozendict(labels)

@dataclass(frozen=False, repr=True, eq=True)
class Sentence:
    raw_text: str # original text, kind-of-sort-of unprocessed. assumed to match slicing the source document with the start/end offsets
    start_offset: int #inclusive, character offset into source text
    end_offset: int #exclusive, characer offset into source text
    tokens: Tuple[Token, ...]
    text: Optional[str] # arbitrarily processed version of raw_text. raw text will be used if this is not defined
    labels: Optional[Dict[str, Any]] # arbitrary collection of labels, user defined and optional

    #def __post_init__(self, text, labels):
    #    if text is None:
    #        self.text = self.raw_text
    #    #if tokens is not None:
    #    #    if type(tokens) is list:
    #    #        self.tokens = tuple(tokens)
    #    assert self.raw_text is not None
    #    if labels is None:
    #        self.labels = None
    #    else:
    #        self.labels = frozendict(labels)

@dataclass(frozen=False, repr=True, eq=True)
class Document:
    docid: str # required
    sentences: Tuple[Sentence, ...]
    raw_text: str # original text, kind-of-sort-of unprocessed. assumed to match slicing the source document with the start/end offsets. You can have done whatever you want to this so long as processing happens before it
    text: Optional[str] # arbitrarily processed version of raw_text. raw text will be used if this is not defined
    #tokenizations: Dict[str, List[Sentence]]: frozendict = frozendict()
    labels: Optional[Dict[str, Any]] # arbitrary collection of labels, user defined and optional
    
    #def __post_init__(self, text, labels):
    #    #if tokenizations is not None and isinstance(tokenization, (dict, frozendict)):
    #    #    self.tokenziations = frozendict(tokenizations)
    #    assert self.docid is not None
    #    assert self.raw_text is not None
    #    assert len(self.sentences) > 0
    #    if text is None:
    #        self.text = self.raw_text
    #    if labels is None:
    #        self.labels = None
    #    else:
    #        self.labels = frozendict(labels)

    def tokens(self) -> Iterable[Token]:
        return chain.from_iterable(s.tokens for s in self.sentences) 

    def sentence_containing(self, index: int) -> Optional[int]:
        # TODO this can be made into a binary search if I ever care
        for i, sent in enumerate(self.sentences):
            if index >= sent.start_offset and index < sent.end_offset:
                return i
        return None

    def sentence_span(self, start: int, end: int) -> Optional[Tuple[int, int]]:
        assert end > start
        assert start >= 0
        ss = self.sentence_containing(start)
        se = self.sentence_containing(end - 1)
        if ss is not None and se is not None:
            return (ss, se + 1)


def to_structured(article: Union[Article, str],
                  fields=None,
                  join_para_on="  ",
                  join_sections_on="\n\n",
                  token_converter: Dict[str, int]=None,
                  pmcid=None) -> Document:
    if isinstance(article, (Article, TextArticle)):
        txt = article.to_raw_str(fields=fields,
                                 join_para_on=join_para_on,
                                 join_sections_on=join_sections_on)
    elif isinstance(article, str):
        txt = article
    else:
        raise ValueError(f'Unknown type {type(article)} for input article')

    doc = nlp(txt)
    sentences = []
    for sent in doc.sents:
        tokens = []
        for token in sent:
            if token_converter:
                token_id = token_convert[token.text]
            else:
                token_id = None
            tokens.append(Token(raw_text=token.text,
                                text=token.text,
                                start_offset=token.pos,
                                end_offset=token.pos + len(token.text),
                                token_id=token_id,
                                labels=None))
        sentences.append(Sentence(raw_text=sent.text,
                                  text=sent.text,
                                  start_offset=sent.start_char,
                                  end_offset=sent.end_char,
                                  tokens=tuple(tokens),
                                  labels=None))
    return Document(docid=pmcid if pmcid is not None else article.get_pmcid(),
                    raw_text=txt,
                    text=txt,
                    sentences=tuple(sentences),
                    labels=None)

def retokenize_with_bert(doc: Document,
                         bert_tokenizer) -> Document:
    vocab = bert_tokenizer.get_vocab()
    sentences = []
    for sent in doc.sentences:
        tokens = []
        for token in sent.tokens:
            wordpieces = bert_tokenizer.tokenize(token.text)
            offset = token.start_offset
            wp_len = 0
            for wp in wordpieces:
                stripped_wp = wp.replace('##', '')
                # TODO should there be any invariants about reconstruction?
                tokens.append(Token(#raw_text=token.text[wp_len:wp_len+len(stripped_wp)],
                                    raw_text=sent.raw_text[sent.start_offset - offset:sent.start_offset - offset + len(stripped_wp)],
                                    text=wp,
                                    start_offset=offset,
                                    end_offset=offset + len(stripped_wp),
                                    token_id=vocab[wp],
                                    labels=token.labels))
                offset += len(stripped_wp)
                wp_len += len(stripped_wp)
        sentences.append(Sentence(raw_text=sent.text,
                                  text=' '.join(t.text for t in tokens),
                                  start_offset=sent.start_offset,
                                  end_offset=sent.end_offset,
                                  tokens=tuple(tokens),
                                  labels=sent.labels))
    return Document(docid=doc.docid,
                    raw_text=doc.raw_text,
                    text=doc.text,
                    sentences=tuple(sentences),
                    labels=doc.labels)

