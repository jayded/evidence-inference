#!/usr/bin/env bash
python evidence_inference/experiments/run_baseline.py --config=heuristic_cheating
python evidence_inference/experiments/run_baseline.py --config=heuristic
python evidence_inference/experiments/run_baseline.py --config=lr
python evidence_inference/experiments/run_baseline.py --config=lr_cheating
python evidence_inference/experiments/run_baseline.py --config=scan_net
python evidence_inference/experiments/run_baseline.py --config=scan_net ICO
python evidence_inference/experiments/run_baseline.py --config=LR_Pipeline