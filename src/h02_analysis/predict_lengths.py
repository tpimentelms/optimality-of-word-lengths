import os
import sys
import argparse
import ast
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--src-fname', type=str, required=True)
    # Model
    parser.add_argument('--language', type=str, required=True, choices=constants.LANGUAGES)
    # Output
    parser.add_argument('--tgt-fname', type=str, required=True)

    return parser.parse_args()


def load_aggregate_results(fname):
    df = pd.read_csv(fname, sep='\t', keep_default_na=False)
    del df['Unnamed: 0']

    return df


def main():
    args = get_args()
    df = load_aggregate_results(args.src_fname)

    # Compute predictions
    df['length'] = df.word.apply(len)
    df['count_smoothed'] = df.train_freq + 1
    df['zipf'] = - np.log2(df.count_smoothed / df.count_smoothed.sum())
    df['zipf_tokenised'] = df.logprob
    df['cch_lower'] = df.mean_surp
    df['squared_surp'] = df.surps.apply(lambda x: np.mean(np.array(ast.literal_eval(x))**2))
    df['cch'] = df.squared_surp / df.mean_surp

    df['length_avg'] = (df.length * df.n_surps).sum() / df.n_surps.sum()

    del df['count_smoothed']
    del df['squared_surp']
    del df['mean_surp']

    # Save results
    df.to_csv(args.tgt_fname, sep='\t')


if __name__ == '__main__':
    main()
