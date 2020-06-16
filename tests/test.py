import unittest
from MLmodels import train_mlmodel, buy_observations
from DataAnalysisTECH import add_extra_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


class FinanceTest(unittest.TestCase):

    def test_otherml_output(self):
        models = [
            {'stock': 'AAPL', 'model': LogisticRegression()},
            {'stock': 'AAPL', 'model': LinearDiscriminantAnalysis()},
            {'stock': 'AAPL', 'model': QuadraticDiscriminantAnalysis()}
            ]

        for model in models:
            self.assertTrue(set(train_mlmodel(model['stock'], model['model'])).
                            issubset(set([-1, 0, 1])),
                            "ML output not expected")

    def test_badmodelinput(self):
        models = [
          {'stock': ' ', 'model': LogisticRegression()},
          {'stock': 'AAPL', 'model': 'Banana'}
              ]
        for model in models:
            with self.assertRaises(Exception):
                train_mlmodel(model['stock'], model['model'])

    def test_buy_zero_observations(self):
        results = [[0, 0, 0],
                   [-1]
                   ]
        for result in results:
            self.assertEqual(buy_observations(result), 0)

    def test_buy_all_observations(self):
        results = [[1, 1, 1],
                   [1]
                   ]
        for result in results:
            self.assertEqual(buy_observations(result), 100.0)

    def test_buy_correct_observations(self):
        results = [[1, 0, 1]
                   ]
        for result in results:
            self.assertEqual(buy_observations(result), 66.667)

    def test_add_extra_features(self):
        High = [51.84, 54.33, 56.52]
        Low = [47.80, 50.06, 54.32]
        Close = [50.4, 54.22, 56.00]
        Open = [47.90, 51.54, 55.21]

        df = pd.DataFrame({'High': pd.Series(High), 'Low': pd.Series(Low),
                          'Close': pd.Series(Close), 'Open': pd.Series(Open)})

        for i in range(len(df)-1):
            self.assertEqual(add_extra_features(df).loc[i, 'IntraDay'],
                             np.around(df.loc[i, 'High'] - df.loc[i, 'Low'], 3))
            self.assertEqual(add_extra_features(df).loc[i, 'PreMarket'],
                             np.around(df.loc[i, 'Close'] - df.loc[i+1, 'Open'], 3))
            self.assertEqual(add_extra_features(df).loc[i, 'Daychange'],
                             np.around((df.loc[i, 'Close'] - df.loc[i, 'Open'])/df.loc[i, 'Open'], 3))


if __name__ == '__main__':
    unittest.main()
