import pandas as pd
from dataclasses import dataclass


@dataclass
class Parameters:
    max_len = 50
    vocab_size = 10000
    embedding_dim = 100

