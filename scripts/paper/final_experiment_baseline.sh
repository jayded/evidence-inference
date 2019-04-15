#!/usr/bin/env bash
cd ../.././
python3 evidence_inference/experiments/run_baseline.py --config=heuristic_cheating
python3 evidence_inference/experiments/run_baseline.py --config=heuristic
python3 evidence_inference/experiments/run_baseline.py --config=lr
python3 evidence_inference/experiments/run_baseline.py --config=lr_cheating
python3 evidence_inference/experiments/run_baseline.py --config=scan_net
python3 evidence_inference/experiments/run_baseline.py --config=scan_net_ICO
python3 evidence_inference/experiments/run_baseline.py --config=LR_Pipeline
