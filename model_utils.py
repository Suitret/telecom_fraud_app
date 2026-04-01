import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class AdaptiveEncoder:
    def __init__(self, column, target, hour_col='is_night_call'):
        self.column = column
        self.target = target
        self.hour_col = hour_col
        self.maps = {}

    def fit(self, df):
        # Target mean encoding
        self.maps['target_mean'] = df.groupby(self.column)[self.target].mean().to_dict()
        # Frequency encoding
        self.maps['freq'] = df[self.column].value_counts(normalize=True).to_dict()
        return self

    def transform(self, df):
        df = df.copy()
        df[f'{self.column}_target'] = df[self.column].map(self.maps['target_mean']).fillna(0)
        df[f'{self.column}_freq'] = df[self.column].map(self.maps['freq']).fillna(0)
        return df
