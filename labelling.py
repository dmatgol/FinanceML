import numpy as np
import pandas as pd
import warnings


def fixed_time_horizon(close, threshold=0, look_forward=1, standardizer=False, window = None):
    """
       Fixed-Time Horizon Labelling Method
       Originally described in the book Advances in Financial Machine Learning, Chapter 3.2, p.43-44.

       Returns 1 if return for a period is greater than the threshold, -1 if less, and 0 if in between. If no threshold is
       provided then it will simply take the sign of the return.

    param close: (pd.Series) Close prices over fixed horizons (usually time bars, but can be any format as long as
                    index is timestamps) for a stock ticker.

    param threshold: (float or pd.Series) When the abs(change) is larger than the threshold, it is labelled as 1 or -1.
                    If change is smaller, it's labelled as 0. Can be dynamic if threshold is pd.Series. If threshold is
                    a series, threshold.index must match close.index. If threshold is negative, then the directionality
                    of the labels will be reversed. If no threshold is given, then the sign of the observation is
                    returned.

    param look_forward: (int) Number of ticks to look forward when calculating future return rate. (1 by default)
                        If n is the numerical value of look_forward, the last n observations will return a label of NaN
                        due to lack of data to calculate the forward return in those cases.

    param standardized: (bool) Whether returns are scaled by mean and standard deviation.

    param window: (int) If standardized is True, the rolling window period for calculating the mean and standard
                    deviation of returns.

    return: (pd.Series) -1, 0, or 1 denoting whether return for each tick is under/between/greater than the threshold.
                    The final look_forward number of observations will be labeled np.nan. Index is same as index of
                    close.
    """
    forward_returns = (close.shift(-look_forward)-close)/close
    print(forward_returns)
    if look_forward >= len(forward_returns):
        warnings.warn('look_forward period is greater than the length of the Series. All Labels will be Nan', UserWarning)

    if standardizer:
        assert isinstance(window, int), "when standardizer is True, window must be an int"
        if look_forward >= len(forward_returns):
            warnings.warn('look_forward period is greater than the length of the Series. All Labels will be Nan', UserWarning)

        mean = forward_returns.rolling(window=window).mean()
        std = forward_returns.rolling(window=window).std()
        forward_returns = (forward_returns-mean)/std

    labels = []
    for forward_return in forward_returns:
        if forward_return > threshold:
            labels.append(1)
        elif ((forward_return <= threshold) & (forward_return >= -threshold)):
            labels.append(0)
        elif forward_return < - threshold:
            labels.append(-1)
        else:
            labels.append(np.nan)
    return pd.Series(labels, index=close.index)
