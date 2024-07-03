import os
import sys
import argparse
import string
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants, utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--src-fname', type=str, required=True)
    parser.add_argument('--tgt-fname', type=str, required=True)
    # Model
    parser.add_argument('--language', type=str, required=True, choices=constants.LANGUAGES)

    return parser.parse_args()


def get_is_clean_f(language):
    alphabet = set(utils.get_alphabet(language))
    def is_clean(x):
        str_chars = set(x)
        return len(alphabet & str_chars) == len(str_chars)

    return is_clean


def read_predictions(fname):
    df = pd.read_csv(fname, sep='\t', keep_default_na=False)
    del df['Unnamed: 0']

    return df


def _train_regression(x, y, weights, fit_intercept=False, use_frequency=True):
    if not use_frequency:
        weights = np.ones_like(y)

    reg = LinearRegression(copy_X=True, fit_intercept=fit_intercept).fit(x, y, weights)
    r_squared = reg.score(x, y, weights)

    y_hat = reg.predict(x)
    mse = (((y - y_hat)**2 * weights).sum() / weights.sum())**.5
    return reg, y_hat, r_squared, mse


def train_regression(x, y, weights, fit_intercept=False, use_frequency=True):
    _, _, r_squared, mse = _train_regression(x, y, weights, fit_intercept, use_frequency)
    return r_squared, mse


def get_expected_cost(x, y, weights, fit_intercept=False, use_frequency=True):
    if not use_frequency:
        weights = np.ones_like(y)

    xnew = x / y[:, np.newaxis]
    ynew = np.ones_like(y)

    reg, y_hat, _, mse = _train_regression(xnew, ynew, weights, fit_intercept, use_frequency)
    cost_normed = mse**2

    capacity = (1 / reg.coef_).item()
    cost = ((xnew[:, 0] - capacity)**2 * weights).sum() / weights.sum()

    assert abs(cost_normed * capacity**2 - cost) < 1e-10
    return cost_normed, cost


def evaluate_predictor(df, predictor, predicted):
    x = df[[predictor]].values
    y = df[predicted].values
    weights = df['n_surps'].values

    r_squared, mse = train_regression(x, y, weights, fit_intercept=False, use_frequency=True)
    cost_normed, cost = get_expected_cost(x, y, weights, fit_intercept=False, use_frequency=True)
    # import ipdb; ipdb.set_trace()
    r_squared_bias, mse_bias = train_regression(x, y, weights, fit_intercept=True, use_frequency=True)
    r_squared_unweighted, mse_unweighted = train_regression(x, y, weights, fit_intercept=False, use_frequency=False)
    r_squared_bias_unweighted, mse_bias_unweighted = train_regression(x, y, weights, fit_intercept=True, use_frequency=False)

    corr_pearson, p_value_pearson = scipy.stats.pearsonr(x[:, 0], y)
    corr_spearman, p_value_spearman = scipy.stats.spearmanr(x[:, 0], y)

    print(f'\t{predictor: <10}. \tR2: {r_squared:.4f}\t\tMSE: {mse:.4f}\t\tCorr: {corr_spearman:.4f} ({p_value_spearman:.4f})')

    return {
        'predictor': predictor,
        'predicted': predicted,
        'n_samples': df.shape[0],
        # 
        'cost_normed': cost_normed,
        'cost': cost,
        'r_squared': r_squared,
        'mse': mse,
        # 
        'corr_spearman': corr_spearman,
        'p_value_spearman': p_value_spearman,
        'corr_pearson': corr_pearson,
        'p_value_pearson': p_value_pearson,
        # 
        'r_squared_bias': r_squared_bias,
        'mse_bias': mse_bias,
        'r_squared_unweighted': r_squared_unweighted,
        'mse_unweighted': mse_unweighted,
        'r_squared_bias_unweighted': r_squared_bias_unweighted,
        'mse_bias_unweighted': mse_bias_unweighted,
    }


def evaluate_predictors(df, clean_method):
    results = []
    df = df.copy()
    df.sort_values('train_freq', ascending=False, inplace=True)

    for max_types in [sys.maxsize, 100000, 50000, 25000, 10000]:
        df_sub = df[:max_types]
        xent = df_sub.xent.unique().item()
        length_avg = df_sub.length_avg.unique().item()
        bpc = df_sub.bpc.unique().item()
        print(f'LinearRegressions (All words, n_samples={df_sub.shape[0]}):')

        # import ipdb; ipdb.set_trace()
        for predictor in ['zipf', 'cch_lower', 'cch', 'zipf_tokenised']:
            for predicted in ['length', 'zipf']:
                result = evaluate_predictor(df_sub.copy(), predictor, predicted)
                result['clean_method'] = clean_method
                result['xent'] = xent
                result['length_avg'] = length_avg
                result['bpc'] = bpc
                result['max_types'] = max_types

                results += [result]

    return results


def main():
    args = get_args()
    df = read_predictions(args.src_fname)

    # Get predictions' correlations
    results = evaluate_predictors(df, clean_method='none')

    # Remove words that contain punctuation
    df = df[~df.word.str.contains('[%s]' % string.punctuation)]
    results += evaluate_predictors(df, clean_method='no_punctuation')

    # Remove words that contain characters not in alphabet
    alphabet = utils.get_alphabet(args.language)
    df = df[~df.word.str.contains('[^%s]' % ''.join(alphabet))]
    results += evaluate_predictors(df, clean_method='alphabet')

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.tgt_fname, sep='\t')


if __name__ == '__main__':
    main()
