import pandas as pd

class SplitDateTime(object):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt_vals = pd.to_datetime(X.Dates)

        def get_vals(dt):
            return pd.Series({
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
                "minute": dt.minute
            })

        parts_df = dt_vals.apply(get_vals)
        return pd.concat([X[["X", "Y"]], parts_df], axis=1)