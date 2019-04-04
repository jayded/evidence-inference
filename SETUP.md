Download embeddings to "embeddings" from
http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin

You should create a virtualenv satisfying requirements.txt. We recommend using
conda. You will need to install a very recent PyTorch (1.0 or later).

Experiments were run on a mix of 1080Tis, K40ms, K80ms, and P100s.
You should be able to (approximately) reproduce the main experiments via the
programs in `scripts/paper/` (you may wish to modify the code to run multiple
trials). The main results should finish in fewer than 10 hours.
