import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from split_date_time import SplitDateTime
from multiclass_log_loss import multiclass_log_loss, multiclass_loss_scorer

result_label_encoder = LabelEncoder()

data = pd.read_csv("data/train.csv")
y = result_label_encoder.fit_transform(data.Category.values)
X = data.drop('Category', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Pipeline uses 8 features and random forest for classifying the different type of crimes
# Features: year, month, day, hour, minute, latitude, longitude and DayOfWeek
# The best parameters for RandomForestClassifier were chosen using grid search
pipe_lr = Pipeline([
    ('dfmapper', DataFrameMapper([
        ('Dates', SplitDateTime()),
        (['X', 'Y'], None),
        ('DayOfWeek', LabelBinarizer())
    ])),
    ('rf', RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=12, max_features=None, n_estimators=80))
])

# Grid Search was used to find the best params for random forest

# param_grid = [{
#     'rf__max_depth': list(range(9, 20)),
#     'rf__n_estimators': list(range(45, 70, 5)),
#     'rf__criterion': ["gini", "entropy"],
#     "rf__max_features": ["auto", None]
# }]
#
# gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring=multiclass_loss_scorer)

# Set estimator to gs to use grid search
estimator = pipe_lr

estimator.fit(X_train, y_train)

train_predicted_probability = estimator.predict_proba(X_train)
train_score = multiclass_log_loss(y_train, train_predicted_probability)

test_predicted_propbability = estimator.predict_proba(X_test)
test_score = multiclass_log_loss(y_test, test_predicted_propbability)
print("Train score: %.3f, Test score: %.3f" % (train_score, test_score))

submission_data = pd.read_csv("data/test.csv")
submission_prob = estimator.predict_proba(submission_data)

label_encoder_columns = result_label_encoder.classes_

probabilities_with_labels = pd.DataFrame(submission_prob)
probabilities_with_labels.columns = label_encoder_columns

submission_df = pd.concat([submission_data.Id, probabilities_with_labels], axis=1)
# Score submitted to kaggle was 2.38866, which was ranked 466 out of 2251 at the time of submission.
submission_df.to_csv("test_predictions/test_prediction1.csv", index=False)
print("Submission csv ready.")