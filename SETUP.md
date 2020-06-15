Download embeddings to "embeddings" from
http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin

You should create a virtualenv satisfying requirements.txt. We recommend using
conda (https://www.anaconda.com/). You will need to install a very recent
PyTorch (1.1 or later):
```
conda create  -p ~/local/evidence_inference_venv
conda activate ~/local/evidence_inference_venv
# this may differ for your CUDA installation
conda install pytorch -c pytorch
pip install -r requirements.txt
python -m spacy download en
```


Experiments were run on a mix of 1080Tis, K40ms, K80ms, and P100s.
You should be able to (approximately) reproduce the main experiments via the
programs in `scripts/paper/` (you may wish to modify the code to run multiple
trials). The main results should finish in fewer than 10 hours.


To run the heuristics, you'll need a mysql server. On Ubuntu, the packages
mysql-server and libmysqlclient-dev should suffice (or equivalent, e.g. Maria).
