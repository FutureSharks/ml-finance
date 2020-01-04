# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def apply(price_series, period=14, method='ema'):
    '''
    Relative strength index
    '''

    delta = price_series.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    if method == 'ema':
        # Calculate the EWMA RSI
        roll_up = up.ewm(com=period, min_periods=period, adjust=True, ignore_na=False).mean()
        roll_down = down.abs().ewm(com=period, min_periods=period, adjust=True, ignore_na=False).mean()
    elif method == 'sma':
        # Calculate the SMA RSI
        roll_up = up.rolling(window=period, min_periods=period, center=False).mean()
        roll_down = down.abs().rolling(window=period, min_periods=period, center=False).mean()

    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))

    return RSI
