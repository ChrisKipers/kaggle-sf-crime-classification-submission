class ColumnExtractor(object):

    def __init__(self, cols):
        self._cols = cols

    def transform(self, X):
        vals = X[self._cols].values
        if len(self._cols) == 1:
            vals.shape = (vals.shape[0],)
        return vals

    def fit(self, X, y=None):
        return self
