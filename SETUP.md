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
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
```

Experiments were run on a mix of 1080Tis, 2080Tis, Titan RTXs. The last is
required when using BiomedRoBERTa instead of SciBERT.

Reproduce the pipelined BERT experiments via:
```
conda activate ~/local/evidence_inference_venv
python evidence_inference/models/pipeline.py --output_dir outputs/ --params params/bert_pipeline_ev2.0.json
```
You may wish to modify or remove training steps as this is rather expensive.

 

For the original paper, you should be able to (approximately) reproduce the main
experiments via the programs in `scripts/paper/` (you may wish to modify the
code to run multiple trials). The main results should finish in fewer than 10
hours.


To run the heuristics, you'll need a mysql server. On Ubuntu, the packages
mysql-server and libmysqlclient-dev should suffice (or equivalent, e.g. Maria).
