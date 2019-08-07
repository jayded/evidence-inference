"""
The function `convert_to_paragraphs` converts a document into a list with one
entry for paragraph, and inclues the rationales for each paragraph.

Usage example (within a Python script):

```
from evidence_inference.preprocess import preprocessor as pp
from evidence_inference.preprocess import paragraphs as ps

prompts = pp.read_prompts()
annotations = pp.read_annotations()

prompt_id = prompts.iloc[0][pp.PROMPT_ID_COL_NAME]
article_id = prompts.iloc[0][pp.STUDY_ID_COL]
article = pp.get_article(article_id)
annots = annotations[annotations[pp.PROMPT_ID_COL_NAME] == prompt_id]
paragraphs = ps.convert_to_paragraphs(article, annots)
```
"""

from collections import namedtuple
import numpy as np

from evidence_inference.preprocess import preprocessor as pp


# Use emojis to separate paragraphs and sections. These shouldn't show up in
# actual article text.
PAR_SEP_CHAR = u"\U0001f604"  # Paragraph separator.
SEC_SEP_CHAR = u"\U0001f605"  # Section separator.

# After the "<p>" is stripped out, we're left with two characters separating
# paragraphs. Initially this was two spaces.
PAR_JOIN = f"{PAR_SEP_CHAR}<p>{PAR_SEP_CHAR}"
PAR_SEP = PAR_SEP_CHAR * 2

# Two characters separating sections. Initially this was two newlines.
SEC_SEP = SEC_SEP_CHAR * 2

Evidence = namedtuple("Evidence", ["start", "end", "text"])


def is_inside(smaller_span, bigger_span):
    """
    Returns True if the smaller span is inside the bigger span. Give a fudge
    factor of two on each side in case the rationale includes paragraph
    boundaries.
    """
    return ((smaller_span[0] >= bigger_span[0] - 2) and
            (smaller_span[1] <= bigger_span[1] + 2))


def check_no_special_seps(msg):
    assert PAR_SEP not in msg
    assert SEC_SEP not in msg


def extract_raw_text_special_seps(article):
    """
    Extract raw text, but with the special paragraph and section separators.
    """
    # Make sure the article doesn't have any separator characters.
    check_no_special_seps(article.get_title())
    check_no_special_seps(article.to_raw_str())

    ti_ab = "TITLE: " + article.get_title() + SEC_SEP
    article_body = article.to_raw_str(join_para_on=PAR_JOIN,
                                      join_sections_on=SEC_SEP)
    raw_text = ti_ab + "  " + article_body

    return raw_text.replace("<p>", "")


def compute_overlap(span1, span2):
    """
    Compute number of overlapping characters in span1 and span2.
    """
    start = max(span1[0], span2[0])
    end = min(span1[1], span2[1])

    return max(0, end - start)


def cleanup_orphans(original_text, paragraphs, evidence_spans, evidence_found):
    """
    Assign rationales that cross paragraphs to the paragraph with the largest
    intersection.
    """
    orphans = evidence_spans - evidence_found
    for orphan in orphans:
        overlaps = [compute_overlap(orphan, paragraph["character_ixs"])
                    for paragraph in paragraphs]
        assert max(overlaps) > 0
        assignment = np.argmax(overlaps)
        orphan_text = original_text[orphan[0]:orphan[1]]
        orphan_evidence = Evidence(orphan[0], orphan[1], orphan_text)
        paragraphs[assignment]["evidence"].append(orphan_evidence)
        evidence_found.add(orphan)

    assert evidence_spans == evidence_found

    return paragraphs


def convert_to_paragraphs(article, annotations):
    """
    Output the article as a list of paragraphs, for use by paragraph-level QA
    systems. Includes info on the section the paragraph came from, and whether
    the paragraph contains any evidence spans.

    `article` is an `article_reader.Article` object.
    `annotations` is a `pd.DataFrame` of the annotations for this document.
    """
    # Get the evidence from the document text.
    evidence_starts = annotations[pp.EVIDENCE_START].values
    evidence_ends = annotations[pp.EVIDENCE_END].values
    evidence_spans = set(zip(evidence_starts, evidence_ends))
    # The end index is inclusive.
    original_text = pp.extract_raw_text(article)
    raw_text = extract_raw_text_special_seps(article)

    # Make sure the special chars version of the text matches original.
    assert raw_text.replace(PAR_SEP_CHAR, " ").replace(SEC_SEP_CHAR, "\n") == original_text

    evidence_found = set()

    sections = raw_text.split(SEC_SEP)
    char_offset = 0
    res = []
    total_paragraphs = 0

    # Loop over sections.
    for section in sections:
        section_name = section.split(":")[0].strip()
        paragraphs = section.split(PAR_SEP)
        # Loop over paragraphs.
        for i, paragraph in enumerate(paragraphs):
            n_chars = len(paragraph)
            # Start and end indices for paragraph.
            ixs = (char_offset, char_offset + n_chars - 1)
            assert paragraph == original_text[ixs[0]:ixs[1] + 1]
            evidence_ixs = [span for span in evidence_spans
                            if is_inside(span, ixs)]
            # Evidence should only be found in one paragraph.
            assert not (evidence_found & set(evidence_ixs))
            evidence_found |= set(evidence_ixs)

            # Assemble evidence.
            par_evidence = []
            for start, end in evidence_ixs:
                evidence_text = original_text[start:end + 1]
                par_evidence.append(Evidence(start, end, evidence_text))

            # Add entry to list of paragraphs.
            entry = dict(paragraph=paragraph,
                         section=section_name,
                         position_in_section=i,
                         position_in_article=total_paragraphs,
                         character_ixs=ixs,
                         evidence=par_evidence)
            res.append(entry)

            # Need to add characters for the stripped-out paragraph separator.
            char_offset += n_chars + len(PAR_SEP)
            total_paragraphs += 1

        # Add characters for the stripped-out section separator. Subtract off
        # the length of the paragraph separator, since it's not used after final
        # paragraph.
        char_offset = char_offset - len(PAR_SEP) + len(SEC_SEP)

    # Some evidence spans cross paragraph boundaries. In that case, assign to
    # the paragraph with most of the text.
    if evidence_spans != evidence_found:
        res = cleanup_orphans(original_text, res, evidence_spans, evidence_found)

    # Final check
    evidences = [x["evidence"] for x in res if x["evidence"]]
    evidences = [x for y in evidences for x in y]
    assert len(evidences) == len(evidence_spans)

    return res
