import sys
import os
import gc
import pickle
import json
import random
import numpy as np
import homoglyphs as hg

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants


def config(seed):
    random.seed(seed)
    np.random.seed(seed)


def string_join(x, j=''):
    return j.join(x)


def write_tsv(df, fname):
    df.to_csv(fname, sep='\t')


def append_json_lines(data, fname):
    with open(fname, 'a', encoding='utf-8') as f:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write('%s\n' % json_str)


def read_json_lines(fname, max_lines=None):
    with open(fname, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data += [json.loads(line)]
            if max_lines and len(data) >= max_lines:
                break

    return data


def read_json_lines_iter(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_json(data, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def read_json(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def get_alphabet(language):
    language = language if language != 'simple' else 'en'

    try:
        alphabet = hg.Languages.get_alphabet([language])
    except ValueError:
        if constants.scripts[language] is not None:
            alphabet = hg.Categories.get_alphabet(
                [x.upper() for x in constants.scripts[language]]
            )
        else:
            raise ValueError

    return alphabet


def mkdir(path):
    return os.makedirs(path, exist_ok=True)


def lsdir(path):
    return next(os.walk(path))[1]


def isfile(fname):
    return os.path.isfile(fname)