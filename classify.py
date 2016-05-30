import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from split_date_time import SplitDateTime
from convert_df_to_dict_list import ConvertDFToDictList
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data/train.csv")
submission = pd.read_csv("data/test.csv")

result_label_encoder = LabelEncoder()
result_encoder = OneHotEncoder(categorical_features=[0])

y = result_label_encoder.fit_transform(data.Category.values)
X = data.drop('Category', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# def format_test_results(results):
#     return pd.concat([test.Id, pd.get_dummies(result_label_encoder.inverse_transform(results))], axis=1)


pipe_lr = Pipeline([
    ('combine features', FeatureUnion([
        ('categorical', Pipeline([
            ('le', SplitDateTime()),
            ('oh', ConvertDFToDictList()),
            ('de', DictVectorizer())
        ]))
    ])),
    ('random forest', RandomForestClassifier(n_jobs=-1, max_depth=4, n_estimators=40))
])

pipe_lr.fit(X_train, y_train)
train_pred = pipe_lr.predict(X_train)
print("Training accuracy: %.3f" % accuracy_score(train_pred, y_train))
test_pred = pipe_lr.predict(X_test)
print("Testing accuracy: %.3f" % accuracy_score(test_pred, y_test))
