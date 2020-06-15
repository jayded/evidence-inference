from os import path
from Levenshtein import distance as string_distance

from evidence_inference.preprocess import preprocessor as pp

DATA_DIR = "./annotations/"

def fix_offsets(ev, i, f, text):
    search_range = 10
    try:
        if ev in text[i-search_range:f+search_range]:
            i = text.index(ev, i-search_range)
            f = i + len(ev)
            return True
    except:
        import pdb; pdb.set_trace()


    min_dist = max(3, len(ev)*0.05)
    min_span = ''
    for i_offset in range(-search_range, search_range):
        for f_offset in range(-search_range, search_range):
            span = text[i + i_offset:f + f_offset]
            dist = string_distance(ev.strip(' '), span.strip(' '))
            if dist <= min_dist:
                min_dist = dist
                min_span = span
                
    if min_span:     
        return True

    return False

annotations = pp.read_annotations()

counter = 0
almost  = 0
exact   = 0
total   = len(annotations)
for _, annot in annotations.iterrows():
    article_file = path.join(DATA_DIR, "txt_files/PMC" + str(annot.PMCID) + ".txt")
    with open(article_file, encoding = 'utf-8') as f:
        text = f.read()
    
    start, end = annot["Evidence Start"], annot["Evidence End"]
        
    raw_text   = text[start:end+1]
    saved_text = pp.extract_raw_text(pp.get_article(annot.PMCID))[start:end + 1]
    counter = counter + 1 if raw_text == saved_text else counter
    if start == end:
        exact   += 1
        almost  += 1
    elif type(annot.Annotations) == str:
        valid   = fix_offsets(annot.Annotations, start, end, text)
        exact   = exact + 1 if saved_text == annot.Annotations else exact
        almost  = almost + 1 if valid else almost
            
print("Number of spans extracted from the XML different from those extracted from the TXT files: {} / {} = {:.2f}".format(counter, total, counter / total))
print("Number of spans extracted from the TXT/XML file that exactly match the ones in the CSV: {} / {} = {:.2f}".format(exact, total, exact / total))
print("Number of spans extracted from the TXT/XML file that almost match the ones in the CSV: {} / {} = {:.2f}".format(almost, total, almost / total))

    