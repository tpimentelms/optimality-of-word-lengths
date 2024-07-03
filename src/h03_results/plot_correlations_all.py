import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants, plot, utils


METRIC_YLABEL = {
    'mse': 'MSE',
    'cost': r'$\mathbb{E}\left[\mathrm{cost}\right]$',
    'cost_normed': r'$\mathbb{E}\left[\mathrm{cost}\right]$',
    'corr_spearman': 'Spearman Correlation',
    'corr_pearson': 'Pearson Correlation',
    'mse_bias': 'MSE (with bias)',
    'mse_unweighted': 'MSE (unweighted)',
    'mse_bias_unweighted': 'MSE (with bias; unweighted)',
}
name_hyp_zipf = r'$\textsc{zipf}$'
name_hyp_lower = r'$\textsc{cch}_{\downarrow}$'
name_hyp_cch = r'$\textsc{cch}$'
name_hyp_zipf_sub = r'$\textsc{zipf}$ (subwords)'


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    return parser.parse_args()


def read_correlations(fname):
    df = pd.read_csv(fname, sep='\t', keep_default_na=False)
    del df['Unnamed: 0']

    return df


def get_langs(checkpoint_dir, dataset):
    path = f'{checkpoint_dir}/{dataset}/'
    return utils.lsdir(path)


def get_tokenizers(checkpoint_dir, dataset, language):
    path = f'{checkpoint_dir}/{dataset}/{language}/'
    return utils.lsdir(path)


def get_n_train_tokens(checkpoint_dir, dataset, language, tokenizer, model):
    path = f'{checkpoint_dir}/{dataset}/{language}/{tokenizer}/{model}/'
    return utils.lsdir(path)


def main():
    args = get_args()
    plot.config_plots()


    predictor_order = {
        'zipf': 1,
        'cch_lower': 2,
        'cch': 3,
        'zipf_tokenised': 4,
    }
    predictor_name_order = {
        name_hyp_zipf: 1,
        name_hyp_lower: 2,
        name_hyp_cch: 3,
        name_hyp_zipf_sub: 4,
    }
    predictor_name = {
        'zipf': name_hyp_zipf,
        'cch_lower': name_hyp_lower,
        'cch': name_hyp_cch,
        'zipf_tokenised': name_hyp_zipf_sub,
    }
    clean_method_order = {
        'none': 1,
        'no_punctuation': 2,
        'alphabet': 3,
    }
    clean_method_name_order = {
        r'$\phi(w) \in \Sigma^{*}$': 1,
        r'$\phi(w) \in \Sigma^{*}_{\circ}$': 2,
        r'$\phi(w) \in \Sigma^{*}_{\alpha}$': 3,
    }
    clean_method_name = {
        # 'none': r'$\emptyset$',
        'none': r'$\phi(w) \in \Sigma^{*}$',
        'no_punctuation': r'$\phi(w) \in \Sigma^{*}_{\circ}$',
        'alphabet': r'$\phi(w) \in \Sigma^{*}_{\alpha}$',
    }

    checkpoint_dir = args.checkpoint_dir
    dataset = args.dataset
    model = args.model

    path = f'{checkpoint_dir}/{dataset}/'
    langs = utils.lsdir(path)

    dfs = []
    for language in get_langs(checkpoint_dir, dataset):
        for tokenizer in get_tokenizers(checkpoint_dir, dataset, language):
            for n_train_tokens in get_n_train_tokens(checkpoint_dir, dataset, language, tokenizer, model):
                fname = f'{checkpoint_dir}/{dataset}/{language}/{tokenizer}/{model}/{n_train_tokens}/correlations.tsv'
                if not utils.isfile(fname):
                    continue

                df = read_correlations(fname)
                df['language'] = language
                df['tokenizer'] = tokenizer
                df['n_train_tokens'] = int(n_train_tokens) if int(n_train_tokens) != -1 else 1e10
                dfs += [df]

    df = pd.concat(dfs)

    # Order categoricals for plots
    df['predictor_name'] = df['predictor'].apply(lambda x: predictor_name[x])
    df['predictor_order'] = df['predictor'].apply(lambda x: predictor_order[x])
    df['clean_method_name'] = df['clean_method'].apply(lambda x: clean_method_name[x])
    df['clean_method_order'] = df['clean_method'].apply(lambda x: clean_method_order[x])

    # Group nsamples for plots
    df['nsamples_max'] = df.groupby(['predicted', 'language', 'tokenizer', 'clean_method', 'n_train_tokens'])['n_samples'].transform(max)
    df['nsamples_normed'] = df.n_samples / df.nsamples_max
    df['nsamples_group'] = df.n_samples
    df.loc[(df.n_samples == df.nsamples_max), 'nsamples_group'] = 'max'

    # Group ntrain for plots
    df['ntrain_max'] = df.groupby(['predicted', 'language', 'tokenizer', 'clean_method'])['n_train_tokens'].transform(max)
    df['ntrain_group'] = df.n_train_tokens
    df.loc[(df.n_train_tokens == df.ntrain_max), 'ntrain_group'] = 'max'
    df['ntrain_group'] = df.ntrain_group.apply(lambda x: int(x / 1e6) if x != 'max' else x)

    df_freq = df[df.predicted == 'zipf'].copy()
    df = df[df.predicted == 'length'].copy()

    plot_main_result(df)
    # plot_tokenizer_result(df)
    # plot_cleanmethod_vs_metrics(df, predictor_name_order, clean_method_name_order)
    # plot_cleanmethod_vs_metrics_crossling(df, predictor_name_order, clean_method_name_order)
    # plot_nsample_vs_metrics(df)
    # plot_nsample_vs_metrics_crossling(df)
    # plot_ntrain_vs_metrics(df)
    # plot_ntrain_vs_metrics_crossling(df)

    # df_freq = df_freq[df_freq.predictor != 'zipf'].copy()
    # plot_ntrain_vs_metrics_crossling(df_freq, fpath='results/freq/')


def plot_cleanmethod_vs_metrics_crossling(df, predictor_order, clean_method_order):
    tokenizer, n_samples, ntrain = 'unigramlm',  25000, 'max'

    df_temp = df[(df.predictor != 'zipf_tokenised') &
                 (df.tokenizer == tokenizer) & 
                 (df.max_types == n_samples) & 
                 (df.ntrain_group == ntrain)].copy()

    predictor_order = {
        name_hyp_zipf: 1,
        name_hyp_lower: 2,
        name_hyp_cch: 3,
    }
    metric_yaxis = {
        'mse': (0, 4),
        'cost': (0, 7),
        'cost_normed': (0, .5),
        'corr_pearson': (-.05, .4),
        'corr_spearman': (-.05, .4),
        'mse_bias': (0, 5.5),
        'mse_unweighted': (0, 5.5),
        'mse_bias_unweighted': (0, 5.5),
    }

    df_temp.sort_values(['language', 'clean_method_order', 'predictor_order'], ascending=[True, True, True], inplace=True)

    for metric in ['mse', 'corr_spearman', 'corr_pearson']:
        fig = plt.figure(figsize=(7.1, 3.2))
        ax = sns.barplot(df_temp, x="clean_method_name", y=metric, hue='predictor_name', order=clean_method_order, hue_order=predictor_order, palette=['C0', 'C1', 'C2', 'C3'])

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(ncols=4, loc='upper center', columnspacing=2, handletextpad=0.5)
        ax.set_ylim(metric_yaxis[metric])
        ax.set_xlabel('')
        ax.set_ylabel(METRIC_YLABEL[metric])

        fpath = 'results/clean_method/'
        utils.mkdir(fpath)
        plt.subplots_adjust(hspace=1, wspace=0.25)
        fig.savefig(f'{fpath}/cleanmethod-crossling__{metric}__tokenizer-{tokenizer}__ntrain-{ntrain}.png', bbox_inches='tight')
        plt.close()

def plot_cleanmethod_vs_metrics(df, predictor_order, clean_method_order):
    tokenizer, n_samples, ntrain = 'unigramlm',  25000, 'max'

    df_temp = df[(df.predictor != 'zipf_tokenised') &
                 (df.tokenizer == tokenizer) & 
                 (df.max_types == n_samples) & 
                 (df.ntrain_group == ntrain)].copy()

    predictor_order = {
        name_hyp_zipf: 1,
        name_hyp_lower: 2,
        name_hyp_cch: 3,
    }

    df_temp.sort_values(['language', 'clean_method_order', 'predictor_order'], ascending=[True, True, True], inplace=True)

    for metric in ['mse', 'corr_spearman', 'corr_pearson']:
        g = sns.FacetGrid(df_temp, col="language", col_wrap=5, height=1.4, aspect=1.5/1.5, legend_out=False)
        g.map(sns.barplot, "clean_method_name", metric, 'predictor_name', order=clean_method_order, hue_order=predictor_order, palette=['C0', 'C1', 'C2', 'C3'],)
        g.add_legend()
        sns.move_legend(g, "lower right", bbox_to_anchor=(.97, 0))
        g.set_titles('{col_name}')
        g.fig.supylabel(METRIC_YLABEL[metric])
        g.set_xlabels('')
        g.set_ylabels('')

        # Rotate xticks
        for ax in g.axes:
            for label in ax.get_xticklabels():
                label.set_rotation(45)

        fpath = 'results/clean_method/'
        utils.mkdir(fpath)
        plt.subplots_adjust(hspace=1, wspace=0.25)
        g.savefig(f'{fpath}/cleanmethod__{metric}__tokenizer-{tokenizer}__ntrain-{ntrain}.png', bbox_inches='tight')
        plt.close()


def plot_main_result(df):
    tokenizer, n_samples, clean_method, ntrain = 'unigramlm',  25000, 'alphabet', 'max'

    df = df[df.predictor != 'zipf'].copy()
    df.loc[(df.predictor == 'zipf_tokenised') & (df.tokenizer == 'rawwords'), 'predictor'] = 'zipf'
    df.loc[(df.predictor == 'zipf') & (df.tokenizer == 'rawwords'), 'tokenizer'] = 'unigramlm'
    df.loc[(df.predictor == 'zipf'), 'predictor_name'] = r'$\textsc{zipf}$'
    df.loc[(df.predictor == 'zipf'), 'predictor_order'] = 1


    df_temp = df[(df.predictor != 'zipf_tokenised') &
                 (df.tokenizer == tokenizer) & 
                 (df.max_types == n_samples) & 
                 (df.clean_method == clean_method) & 
                 (df.ntrain_group == ntrain)].copy()

    df_temp.sort_values(['language', 'predictor_order'], ascending=[True, True], inplace=True)
    _plot_main_result(df_temp, tokenizer, clean_method, ntrain, folder='main')


def _plot_main_result(df, tokenizer, clean_method, ntrain, folder):
    metric_yaxis = {
        'mse': (0, 5),
        'cost': (0, 7),
        'cost_normed': (0, .5),
        'corr_pearson': (-.1, .5),
        'corr_spearman': (-.1, .5),
        'mse_bias': (0, 5.5),
        'mse_unweighted': (0, 5.5),
        'mse_bias_unweighted': (0, 5.5),
    }
    metric_figsize = {
        'mse': (7.1, 3.2),
        'cost': (12, 3),
        'cost_normed': (12, 3),
        'corr_pearson': (7.1, 3.2),
        'corr_spearman': (7.1, 3.2),
        'mse_bias': (7.1, 3.2),
        'mse_unweighted': (7.1, 3.2),
        'mse_bias_unweighted': (7.1, 3.2),
    }

    for metric in ['mse', 'corr_spearman', 'corr_pearson']:
        fig = plt.figure(figsize=metric_figsize[metric])
        ax = sns.barplot(df, x="language", y=metric, hue='predictor_name')
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(ncols=4, loc='upper center', columnspacing=2, handletextpad=0.5)
        ax.set_ylim(metric_yaxis[metric])
        ax.set_xlabel('')
        ax.set_ylabel(METRIC_YLABEL[metric])
        
        fpath = f'results/{folder}/'
        utils.mkdir(fpath)
        fig.savefig(f'{fpath}/{folder}__{metric}__tokenizer-{tokenizer}__clean_method-{clean_method}__ntrain-{ntrain}.png', bbox_inches='tight')
        plt.close()


def plot_tokenizer_result(df):
    tokenizer, n_samples, clean_method, ntrain = 'rawwords',  25000, 'alphabet', 'max'

    df_temp = df[(df.predictor != 'zipf_tokenised') &
                 (df.tokenizer == tokenizer) & 
                 (df.max_types == n_samples) & 
                 (df.clean_method == clean_method) & 
                 (df.ntrain_group == ntrain)].copy()

    df_temp.sort_values(['language', 'predictor_order'], ascending=[True, True], inplace=True)
    df_temp = pd.melt(df_temp, id_vars=['predictor_name', 'language'], value_vars=['mse', 'corr_spearman'],
                       var_name='metric', value_name='value')

    df_temp['metric_names'] = df_temp.metric.apply(lambda x: METRIC_YLABEL[x])
    _plot_tokenizer_result(df_temp, tokenizer, clean_method, ntrain, folder='tokenizer', keeplegend=False)

    tokenizer, n_samples, clean_method, ntrain = 'unigramlm',  25000, 'alphabet', 'max'
    df_temp = df[(df.tokenizer == tokenizer) & 
                 (df.max_types == n_samples) & 
                 (df.clean_method == clean_method) & 
                 (df.ntrain_group == ntrain)].copy()
    df_temp.sort_values(['language', 'predictor_order'], ascending=[True, True], inplace=True)

    df_temp = pd.melt(df_temp, id_vars=['predictor_name', 'language'], value_vars=['mse', 'corr_spearman'],
                       var_name='metric', value_name='value')
    df_temp['metric_names'] = df_temp.metric.apply(lambda x: METRIC_YLABEL[x])
    _plot_tokenizer_result(df_temp, tokenizer, clean_method, ntrain, folder='tokenizer')


def _plot_tokenizer_result(df, tokenizer, clean_method, ntrain, folder, keeplegend=True):
    g = sns.FacetGrid(df, col="metric_names", col_wrap=3, height=3.2/1.5, aspect=1.8*1.5, legend_out=False, sharey=False)
    g.map(sns.barplot, "language", 'value', 'predictor_name', palette=['C0', 'C1', 'C2', 'C3'])
    if keeplegend:
        g.add_legend()
        sns.move_legend(g, "lower center", bbox_to_anchor=(.36, -.1), ncols=4)

    g.set_titles('{col_name}')
    g.set_xlabels('')
    g.set_ylabels('')

    fpath = f'results/{folder}/'
    utils.mkdir(fpath)
    g.savefig(f'{fpath}/{folder}__tokenizer-{tokenizer}__clean_method-{clean_method}__ntrain-{ntrain}.png', bbox_inches='tight')
    plt.close()


def plot_nsample_vs_metrics_crossling(df):
    tokenizer, clean_method, ntrain = 'unigramlm', 'alphabet', 'max'
    df_temp = df[(df.predictor != 'zipf_tokenised') &
                 (df.tokenizer == tokenizer) & 
                 (df.clean_method == clean_method) & 
                 (df.ntrain_group == ntrain)].copy()

    df_temp.sort_values(['language', 'n_samples', 'predictor_order'], ascending=[True, True, True], inplace=True)

    metric_yaxis = {
        'mse': (1, 4),
        'cost': (0, 7),
        'cost_normed': (0, .5),
        'corr_pearson': (-.05, .4),
        'corr_spearman': (-.05, .4),
        'mse_bias': (0, 5.5),
        'mse_unweighted': (0, 5.5),
        'mse_bias_unweighted': (0, 5.5),
    }

    df_temp.loc[df_temp.nsamples_group == 'max', 'n_samples'] = 2e5

    for metric in ['mse', 'corr_spearman', 'corr_pearson']:
        fig = plt.figure(figsize=(7.1, 3.2))
        ax = sns.lineplot(df_temp, x="n_samples", y=metric, hue='predictor_name', palette=['C0', 'C1', 'C2'], linewidth=2)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(ncols=4, loc='upper center', columnspacing=2, handletextpad=0.5)
        ax.set_ylim(metric_yaxis[metric])
        ax.set_xlabel(r'\# Word Types')
        ax.set_ylabel(METRIC_YLABEL[metric])

        fpath = 'results/ntypes/'
        utils.mkdir(fpath)
        plt.subplots_adjust(hspace=1, wspace=0.25)
        fig.savefig(f'{fpath}/ntypes-crosling__{metric}__tokenizer-{tokenizer}__clean_method-{clean_method}__ntrain-{ntrain}.png', bbox_inches='tight')
        plt.close()

def plot_nsample_vs_metrics(df):
    tokenizer, clean_method, ntrain = 'unigramlm', 'alphabet', 'max'
    df_temp = df[(df.predictor != 'zipf_tokenised') &
                 (df.tokenizer == tokenizer) & 
                 (df.clean_method == clean_method) & 
                 (df.ntrain_group == ntrain)].copy()

    df_temp.sort_values(['language', 'n_samples', 'predictor_order'], ascending=[True, True, True], inplace=True)

    for metric in ['mse', 'corr_spearman', 'corr_pearson']:
        g = sns.FacetGrid(df_temp, col="language", col_wrap=5, height=1.4, aspect=1.5/1.5, legend_out=False)
        g.map(sns.lineplot, "n_samples", metric, 'predictor_name', sizes=(2,2.1), palette=['C0', 'C1', 'C2'])
        g.set(xscale='log')
        g.add_legend()

        sns.move_legend(g, "lower right", bbox_to_anchor=(1, 0.1))
        g.fig.supylabel(METRIC_YLABEL[metric])
        g.set_titles('{col_name}')
        g.set_xlabels('')
        g.set_ylabels('')
        fpath = 'results/ntypes/'
        utils.mkdir(fpath)
        plt.subplots_adjust(hspace=1, wspace=0.25)
        g.savefig(f'{fpath}/ntypes__{metric}__tokenizer-{tokenizer}__clean_method-{clean_method}__ntrain-{ntrain}.png', bbox_inches='tight')
        plt.close()


def plot_ntrain_vs_metrics_crossling(df, fpath='results'):
    tokenizer, clean_method, n_samples = 'unigramlm', 'alphabet', 25000
    df_temp = df[(df.predictor != 'zipf_tokenised') &
                 (df.max_types == n_samples) & 
                 (df.tokenizer == tokenizer) & 
                 (df.clean_method == clean_method)].copy()

    if fpath == 'results':
        metric_yaxis = {
            'mse': (0.5, 4),
            'cost': (0, 7),
            'cost_normed': (0, .5),
            'corr_pearson': (-.05, .4),
            'corr_spearman': (-.05, .4),
            'mse_bias': (0, 5.5),
            'mse_unweighted': (0, 5.5),
            'mse_bias_unweighted': (0, 5.5),
        }
    else:
        metric_yaxis = {
            'mse': (0.5, 5),
            'cost': (0, 7),
            'cost_normed': (0, .5),
            'corr_pearson': (-.05, 1),
            'corr_spearman': (.2, 1),
            'mse_bias': (0, 5.5),
            'mse_unweighted': (0, 5.5),
            'mse_bias_unweighted': (0, 5.5),
        }

    df_temp.sort_values(['language', 'n_train_tokens', 'predictor_order', 'clean_method_order'], ascending=[True, True, True, True], inplace=True)

    for metric in ['mse', 'corr_spearman', 'corr_pearson']:
        fig = sns.lmplot(df_temp, x="xent", y=metric, hue='predictor_name', palette=['C0', 'C1', 'C2'], aspect=7.1/4.2, height=4.2, legend_out=False)
        ax = plt.gca()

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(ncols=4, loc='upper center', columnspacing=2, handletextpad=0.5)
        ax.set_ylim(metric_yaxis[metric])
        ax.set_xlabel('Language Model\'s Cross-entropy')
        ax.set_ylabel(METRIC_YLABEL[metric])
        save_fig(fig, f'{fpath}/xent/', f'xent-crosling__{metric}__tokenizer-{tokenizer}__clean_method-{clean_method}__nsamples-{n_samples}.png')


def plot_ntrain_vs_metrics(df, fpath='results'):
    tokenizer, clean_method, n_samples = 'unigramlm', 'alphabet', 25000
    df_temp = df[(df.predictor != 'zipf_tokenised') &
                 (df.max_types == n_samples) & 
                 (df.tokenizer == tokenizer) & 
                 (df.clean_method == clean_method)].copy()

    df_temp.sort_values(['language', 'n_train_tokens', 'predictor_order', 'clean_method_order'], ascending=[True, True, True, True], inplace=True)

    for metric in ['mse', 'corr_spearman', 'corr_pearson']:
        g = sns.FacetGrid(df_temp, col="language", col_wrap=5, height=1.4, aspect=1.5/1.5, legend_out=False)

        g.map(sns.lineplot, "n_train_tokens", metric, 'predictor_name')
        g.set(xscale='log')
        g.add_legend()
        sns.move_legend(g, "lower right", bbox_to_anchor=(.98, 0.1))

        g.fig.supylabel(METRIC_YLABEL[metric])
        g.fig.supxlabel('\# Train Tokens')
        g.set_titles('{col_name}')
        g.set_xlabels('')
        g.set_ylabels('')
        plt.subplots_adjust(hspace=1, wspace=0.25)
        save_fig(g, f'{fpath}/ntrain/', f'ntrain__{metric}__tokenizer-{tokenizer}__clean_method-{clean_method}__nsamples-{n_samples}.png')

        g = sns.FacetGrid(df_temp, col="language", col_wrap=5, height=1.7, aspect=1.5/1.8, legend_out=False, sharex=False)
        g.map(sns.lineplot, "xent", metric, 'predictor_name')
        g.add_legend()
        sns.move_legend(g, "lower right", bbox_to_anchor=(.98, 0.1))
        g.fig.supylabel(METRIC_YLABEL[metric])
        g.fig.supxlabel('Language Model\'s Cross-entropy')
        g.set_titles('{col_name}')
        g.set_xlabels('')
        g.set_ylabels('')
        for ax in g.axes:
            for label in ax.get_xticklabels():
                label.set_rotation(30)
        plt.subplots_adjust(hspace=1, wspace=0.25)
        save_fig(g, f'{fpath}/xent/', f'xent__{metric}__tokenizer-{tokenizer}__clean_method-{clean_method}__nsamples-{n_samples}.png')


def save_fig(fig, fpath, fname):
    utils.mkdir(fpath)
    fig.savefig(f'{fpath}/{fname}', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
