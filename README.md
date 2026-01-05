# Scientific-Fraud-Detection

### IMPORTANT UPDATE ###

This repository contains the **corrected implementation** of the system described in the 2024 paper "Detecting Scientific Fraud Using Argument Mining". The original results were invalid due to an error found by independent researchers. This updated codebase fixes those issues.

## Changes

The old arugment quality data (GAQ corpus) is now defunct (is no longer available), so this codevase introduces a new synthetic data generation script, and the updated results are based on this:

  Fraudulent accuracy (Recall/Sensitivity): 0.9640
  Legitimate accuracy (Specificity): 0.8744
  Overall accuracy: 0.9151
  Precision: 0.8645
  Recall: 0.9640
  F1 Score: 0.9116
  
---

## Overview

Code for the paper: 'Detecting Scientific Fraud Using Argument Mining'. All code contained is for illustrative purposes and not intended for use in production settings.

## Usage

- `models/argumentation_based_fraud_detection.py` - Contains the entire fraud detection system
- `models/argument_mining_only.py` - Contains the argument mining component in isolation
- `gpt5_classifier.py` - GPT-5 based classifier implementation

Run the model files from inside the `models` directory.

## Data

Training and evaluation data can be found in the `data/` directory:
- `fraudulent_abstracts.json` - Fraudulent paper abstracts
- `legitimate_abstracts.json` - Legitimate paper abstracts
- `SciARK.json` - SciARK dataset for argument mining

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```
