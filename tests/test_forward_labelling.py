import unittest
import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)) + '/DataAnalysisML')
import labelling as lb


class TestFixedTimeLabelling(unittest.TestCase):
    """Test fixed time horizon labelling method"""

    def setUp(self):
        """
        Set up the file path for the time data
        """
        project_path = os.path.dirname(os.path.dirname(__file__))
        self.path = project_path + '/stock_dfs/AAPL.csv'
        self.data = pd.read_csv(self.path, index_col='Date')
        self.idx10 = self.data[-10:].index

    def test_basic(self):
        """Test for constant threshold and no standardization"""

        close = self.data['Adj Close'][-10:]
        test1 = lb.fixed_time_horizon(close, 0.01, look_forward=1)
        test2 = lb.fixed_time_horizon(close, 0, look_forward=2)
        test3 = lb.fixed_time_horizon(close, 0.004, look_forward=1)
        test4 = lb.fixed_time_horizon(close, 1, look_forward=3)
        test1_actual = pd.Series([0, 0, 0, 1, 0, 1, 1, -1, 0, np.nan], index=self.idx10)
        test2_actual = pd.Series([1, -1, 1, 1, 1, 1, -1, -1, np.nan, np.nan], index=self.idx10)
        test3_actual = pd.Series([1, 1, -1, 1, 1, 1, 1, -1, 1, np.nan], index=self.idx10)
        test4_actual = pd.Series([0, 0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan], index=self.idx10)
        pd.testing.assert_series_equal(test1_actual, test1)
        pd.testing.assert_series_equal(test2_actual, test2)
        pd.testing.assert_series_equal(test3_actual, test3)
        pd.testing.assert_series_equal(test4_actual, test4)

    def test_dynamic_threshold(self):
        """Test the threshold as a pandas Series rather than a constant"""

        close = self.data['Adj Close'][-10:]
        threshold1 = pd.Series([0.01, 0.005, 0, 0.01, 0.02, 0.03, 0.1, -1, 0.99, 0], index=self.idx10)

        test5 = lb.fixed_time_horizon(close, threshold1, look_forward=1)
        test6 = lb.fixed_time_horizon(close, threshold1, look_forward=3)
        test5_actual = pd.Series([0, 1, -1, 1, 0, 1, 0, 1, 0,  np.nan], index = self.idx10)
        test6_actual = pd.Series([0, 1, 1, 1, 1, 0, 0, np.nan, np.nan, np.nan], index = self.idx10)
        pd.testing.assert_series_equal(test5_actual, test5)
        pd.testing.assert_series_equal(test6_actual, test6)

    def test_with_standardization(self):
        """
        Test cases with standardization, with constant and dynamic threshold
        """

        close = self.data['Adj Close'][-10:]
        threshold2 = pd.Series([1, 2, 0, 0.01, 1.5, 15, 300, -1, 0.99, 1], index=self.idx10)

        test7 = lb.fixed_time_horizon(close, 1, look_forward=1, standardizer=True, window=4)
        test8 = lb.fixed_time_horizon(close, 1, look_forward=1, standardizer=True, window=3)
        test9 = lb.fixed_time_horizon(close, threshold2, look_forward=2, standardizer=True, window=5)
        test7_actual = pd.Series([np.nan, np.nan, np.nan, 1, 0, 0, 0, -1, 0, np.nan], index=self.idx10)
        test8_actual = pd.Series([np.nan, np.nan, -1, 1, 0, 0, 0, -1, 0, np.nan], index=self.idx10)
        test9_actual = pd.Series([np.nan, np.nan, np.nan, np.nan, 0, 0, 0, -1, np.nan, np.nan], index=self.idx10)
        pd.testing.assert_series_equal(test7_actual, test7)
        pd.testing.assert_series_equal(test8_actual, test8)
        pd.testing.assert_series_equal(test9_actual, test9)

    def test_look_forward_warning(self):
        """Check if the correct warning is raised when the look_forward parameter is greater
            than the series length
        """
        close = self.data['Adj Close'][-10:]
        with self.assertWarns(UserWarning):
            labels = lb.fixed_time_horizon(close, look_forward=50)
        np.testing.assert_allclose(labels, [np.nan] * len(close))

    def test_look_forward_standardized_warning(self):
        """
        Check if the correct warning is raised when the look_forward parameter is greater
        than the series length.
        Check if an exception is raised when the standardizer parameter is set to True, but
        window is not specified.
        """
        close = self.data['Adj Close'][-10:]
        with self.assertWarns(UserWarning):
            labels = lb.fixed_time_horizon(close, look_forward=50, standardizer=True, window=4)
        np.testing.assert_allclose(labels, [np.nan] * len(close))

        with self.assertRaises(Exception):
            fixed_time_horizon(close, 0, look_forward=50, standardizer=True)

if __name__ == '__main__':
    unittest.main()
