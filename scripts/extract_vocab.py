from gensim.models import KeyedVectors
path_to_wvs = "embeddings/PubMed-w2v.bin"
WVs = KeyedVectors.load_word2vec_format(path_to_wvs, binary=True)
with open("annotations/vocab.txt", 'w') as vf:
    # note the value here has both a count and the number of times the token appeared
    for w, _ in WVs.vocab.items():
        vf.write("{}\n".format(w))
