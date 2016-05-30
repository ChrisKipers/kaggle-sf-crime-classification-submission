class ConvertDFToDictList(object):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.T.to_dict().values()