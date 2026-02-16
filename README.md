# Sensorimotor Norm Imputation (sBERT & GPT-4o)

This repository contains code used to estimate missing sensorimotor norms for a our TED Talks dataset (Park et al.,2016).

Background

We use the Lancaster Sensorimotor Norms (Lynott et al., 2020) dataset (~40k words) as ground truth.
Some words in the TED dataset (~0.82%) were missing ratings — mostly proper nouns.

solution tested:

GPT-4o-mini prompt-based estimation
Prompted using Lancaster definitions
Temperature = 0
Explicit instruction to compute Minkowski3
This repo contains the GPT-based estimation workflow.

Reproducibility

The workflow:
Load Lancaster norms
Define missing words
Provide structured prompt including:
Definition of each modality
Definition of Minkowski3 (exponent 3)
Request numeric ratings (0–5)
Extract token usage

Setup
1. Clone the repo
git clone https://github.com/shrishasat/sensorimotor-gpt-sbert-imputation.git
cd sensorimotor-gpt-sbert-imputation
2. Install dependencies
pip install -r requirements.txt
3. Add your API key

Create a .env file:

AZURE_OPENAI_KEY=your_actual_key_here
Run
python gpt_estimation.py

Notes
Temperature is fixed at 0.0 for determinism.
Future evaluation should assess:
Stability across repeated runs

Correlation with held-out human ratings

Comparison with sBERT regression estimates
