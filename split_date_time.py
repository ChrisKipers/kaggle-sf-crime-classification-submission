import pandas as pd
import numpy as np

class SplitDateTime(object):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt_vals = pd.to_datetime(X)

        def get_vals(dt):
            return [
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute
            ]

        split_dates = [get_vals(x) for x in dt_vals]
        return np.matrix(split_dates)