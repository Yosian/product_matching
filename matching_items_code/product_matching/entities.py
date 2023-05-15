import pandas as pd
from dataclasses import dataclass


@dataclass
class Parameters:
    max_len = 50
    vocab_size = 10000
    embedding_dim = 100


def import_data():
    temp = pd.read_csv('tests/train.tsv', sep='\t', index_col='train_id', nrows=400000,
                       usecols=['train_id', 'name', 'item_description'])
    return temp.astype(str)
