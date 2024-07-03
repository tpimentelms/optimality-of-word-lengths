import os
import sys
import math
import argparse
import pandas as pd


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants, utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--src-fname-surps', type=str, required=True)
    parser.add_argument('--src-fname-freqs', type=str, required=True)
    parser.add_argument('--src-fname-stats', type=str, required=True)
    parser.add_argument('--tgt-fname', type=str, required=True)

    return parser.parse_args()


def read_surprisals(src_fname):
    df = pd.read_csv(src_fname, sep='\t', keep_default_na=False)
    del df['Unnamed: 0']
    df['score'] = df['score'] / math.log(2)

    df = df.groupby('word')['score'].agg([list, 'mean', 'count']).reset_index()
    df.rename(columns={'mean': 'mean_surp', 'list': 'surps', 'count': 'n_surps'}, inplace=True)

    df = df[df.word != '']
    remove_words = ['</s>', '<unk>', '\n']
    for word in remove_words:
        df = df[~df.word.str.contains(word, regex=False)]
    return df


def read_stats(src_fname):
    df = pd.read_csv(src_fname, sep='\t', keep_default_na=False)

    xent = df['xent'].item() / math.log(2)
    surp_words = df['surp_words'].item()
    test_chars = df['test_chars'].item()

    bpc = xent * surp_words / test_chars

    return xent, bpc


def read_frequencies(src_fname):
    df = pd.read_csv(src_fname, sep='\t', keep_default_na=False)
    del df['Unnamed: 0']

    df.word = df.word.apply(lambda x: x.strip())
    df = df.groupby('word').agg({'test_freq': 'sum', 'train_freq': 'sum', 
                                 'val_freq': 'sum', 'logprob': 'min'})
    return df


def main():
    args = get_args()
    df_surps = read_surprisals(args.src_fname_surps)
    df_freqs = read_frequencies(args.src_fname_freqs)
    xent, bpc = read_stats(args.src_fname_stats)

    df = df_surps.join(df_freqs, on='word', how='inner')
    df['xent'] = xent
    df['bpc'] = bpc

    df.to_csv(args.tgt_fname, sep='\t')


if __name__ == '__main__':
    main()
