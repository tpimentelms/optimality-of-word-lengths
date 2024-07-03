# optimality-of-word-lengths


This repository accompanies the paper "Revisiting the Optimality of Work Lengths"

## Install Dependencies

First, create a conda environment with
```bash
$ conda env create -f scripts/environment.yml
```

## Get Data

Extract surprisals for analysed languages using the language model of your choice. In our paper, we trained models using repository [tpimentelms/fairseq-lm-train-and-eval](https://github.com/tpimentelms/fairseq-lm-train-and-eval).
Put a `surprisals.tsv` file with results in path `data/wiki40b/<language>/bpe/language_modeling/transformer/<max_train_tokens>/`; this file must contain columns: `word`, `score`.
Put a `freq_tokens.tsv` file with results in same path; this file must contain columns: `word`, `logprob`.

## Run Analysis

Get  the predictions for a language using command:
```bash
$ make LANGUAGE=<language> N_MODEL_TRAIN_TOKENS=<max_train_tokens>
```

After running the command above for all languages and training token sizes, use the followign command to produce all plots:
```bash
$ make plot_correlations
```
Some plotting functions are commented out, but just uncomment them in script `src/h03_analysis/plot_correlations_all.py`.
