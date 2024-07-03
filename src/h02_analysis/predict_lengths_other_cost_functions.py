import os
import sys
import argparse
import ast
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants, utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--src-fname', type=str, required=True)
    # Output
    parser.add_argument('--tgt-fname', type=str, required=True)

    return parser.parse_args()


def load_aggregate_results(fname):
    df = pd.read_csv(fname, sep='\t', keep_default_na=False)
    del df['Unnamed: 0']

    return df


class FeaturePredictor(nn.Module):
    name = 'FeaturePredictor'

    def __init__(self, n_tokens, loss_fn):
        super().__init__()

        self.n_tokens = n_tokens
        self.loss_fn = loss_fn

        # Create lengths per token array
        indexes = []
        for i, n_tokens_per_type in enumerate(n_tokens):
            indexes += [i] * n_tokens_per_type
        indexes = np.array(indexes)
        self.index = nn.Parameter(torch.from_numpy(indexes), requires_grad=False)

    def init_optimizer(self, params):
        # Optimizer
        self.optimizer = optim.AdamW([params], lr=0.01)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1)

    @property
    def lengths_tokens(self):
        return self.lengths[self.index]

    def forward(self, x):
        inforate = x / self.lengths_tokens
        return inforate

    def criterion(self, inforate):
        return self.loss_fn(inforate, self.capacity)

    def train_batch(self, x):
        self.optimizer.zero_grad()
        inforate = self(x)
        loss = self.criterion(inforate).mean()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, x, nupdates=50000):
        pbar = tqdm(range(nupdates), desc=f'Optimizing the {self.name}', file=sys.stdout)
        print_interval, running_loss, best_loss, waiting_updates = 100, float('inf'), float('inf'), 0
        for i in pbar:
            loss = self.train_batch(x)
            running_loss += loss / print_interval

            if (i % print_interval) == 0:
                pbar.set_description(f'Optimizing the {self.name}. Loss {loss:.5f}. Best loss: {best_loss:.5f}. Wait: {waiting_updates}')
                if (best_loss - running_loss) < 1e-4 and (waiting_updates > 3):
                    break
                elif (best_loss - running_loss) < 1e-4:
                    waiting_updates += 1
                else:
                    waiting_updates = 0

                best_loss, running_loss = min(best_loss, running_loss), 0
                self.scheduler.step()


class CapacityPredictor(FeaturePredictor):
    name = 'CapacityPredictor'

    def __init__(self, lengths, n_tokens, loss_fn):
        super().__init__(n_tokens, loss_fn)

        # Create lengths per type array
        self.lengths = nn.Parameter(torch.from_numpy(lengths), requires_grad=False)
        # Capacity
        self.capacity = nn.Parameter(torch.ones(1))

        self.init_optimizer(self.capacity)


class LengthPredictor(FeaturePredictor):
    name = 'LengthPredictor'

    def __init__(self, lengths_init, capacity, n_tokens, loss_fn):
        super().__init__(n_tokens, loss_fn)

        # Create lengths per type array
        self.lengths = nn.Parameter(torch.from_numpy(lengths_init))

        # Capacity
        self.capacity = nn.Parameter(torch.Tensor([capacity]), requires_grad=False)

        self.init_optimizer(self.lengths)


def get_foo_quadratic_cost(alpha):
    def cost_quadratic_generalised(x, y):
        error = x - y
        error_quadratic = error**2
        error_quadratic[error > 0] = alpha * error_quadratic[error > 0]
        return error_quadratic
    return cost_quadratic_generalised


def get_optimal_preds(df, cost_foo, name):
    # Explode data to instances per-surprisal
    df_exploded = df[['surps', 'length']].explode('surps')
    x = torch.from_numpy(df_exploded.surps.astype(float).values)
    y = torch.from_numpy(df_exploded.length.values)

    # Get optimal capacity
    capacity_predictor = CapacityPredictor(lengths=df.length.values, n_tokens=df.n_surps.values, loss_fn=cost_foo)
    capacity_predictor.fit(x)

    # Get optimal lengths
    capacity = capacity_predictor.capacity.item()
    lengths_init = df.uid.values / capacity
    length_predictor = LengthPredictor(lengths_init=lengths_init, capacity=capacity, n_tokens=df.n_surps.values, loss_fn=cost_foo)
    length_predictor.fit(x)

    # Get final loss
    inforate = length_predictor(x)
    loss_optimal = length_predictor.criterion(inforate).mean()
    loss_real = length_predictor.criterion(x / y).mean()

    # Save results 
    df[name + '-lengths'] = length_predictor.lengths.detach().cpu().numpy()
    df[name + '-capacity'] = capacity_predictor.capacity.detach().cpu().numpy().item()
    df[name + '-loss'] = loss_optimal.detach().cpu().numpy().item()
    df[name + '-loss_real'] = loss_real.detach().cpu().numpy().item()


def save_results(df, tgt_fname):
    df_temp = df.copy()
    del df_temp['surps']
    df_temp.to_csv(tgt_fname, sep='\t')


def load_old_results(df, tgt_fname):
    done_alphas = []
    if utils.isfile(tgt_fname):
        df_temp = load_aggregate_results(tgt_fname)
        regex = lambda x: re.findall(r'cost_quadratic-(\d+\.\d+)-capacity', x)
        done_alphas = [float(regex(x)[0]) for x in df_temp.columns if regex(x)]

        for alpha in done_alphas:
            for infocols in ['lengths', 'capacity', 'loss', 'loss_real']:
                df[f'cost_quadratic-{alpha}-{infocols}'] = df_temp[f'cost_quadratic-{alpha}-{infocols}']

    return done_alphas


def main():
    args = get_args()

    df = load_aggregate_results(args.src_fname)

    # Compute predictions
    df['length'] = df.word.apply(len)
    df['surps'] = df.surps.apply(lambda x: np.array(ast.literal_eval(x)))

    df['squared_surp'] = df.surps.apply(lambda x: np.mean(x**2))
    df['mean_surp'] = df.surps.apply(lambda x: np.mean(x))
    df['uid'] = df.squared_surp / df.mean_surp

    del df['squared_surp']
    del df['mean_surp']

    done_alphas = load_old_results(df, args.tgt_fname)

    for alpha in tqdm(np.arange(1, 5, .25), desc='Optimising alphas', file=sys.stdout):
        tqdm.write(f'Processing alpha: {alpha}')
        if alpha in done_alphas:
            continue

        cost_foo = get_foo_quadratic_cost(alpha)
        get_optimal_preds(df, cost_foo, name=f'cost_quadratic-{alpha}')
        tqdm.write('')

        save_results(df, args.tgt_fname)
    save_results(df, args.tgt_fname)


if __name__ == '__main__':
    main()
