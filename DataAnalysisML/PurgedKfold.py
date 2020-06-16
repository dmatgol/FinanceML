
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.model_selection import BaseCrossValidator
import labelling as lb


class PurgedKFold(KFold):
    """
    Extends Kfold to work with labels that span intervals.
    The train is purged of observations that overlap test-label observations.
    Test set is assumed to be contiguous, w/o train examples in between.
    """

    def __init__(self, n_splits=3, time_idx=None, pctEmbargo=0):
        if not isinstance(time_idx, pd.Series):
            raise ValueError('Label Dates must be a pandas Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.time_idx = time_idx
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index==self.time_idx.index).sum() != len(self.time_idx):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pctEmbargo)
        test_ranges = [(ix[0], ix[-1]+1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        print(X.iloc[:test_ranges[0][1]])
        for i, j in test_ranges:
            test_start = self.time_idx.index[i]
            test_indices = indices[i:j]
            init_train_idx = self.time_idx.index.searchsorted(self.time_idx[test_indices].max())
            train_indices = self.time_idx.index.searchsorted(self.time_idx[self.time_idx.index<test_start].index)
            train_indices = np.concatenate((train_indices, indices[init_train_idx+embargo:]))
            print(train_indices, test_indices)

df = pd.read_csv('Aaple_test.csv')
time_df = df['Date']
#print(time_df)
PurgedKFold(time_idx=time_df).split(df, lb.fixed_time_horizon(df['Adj Close']))
