import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_val_score
import numpy as np

# load data
features_w2v2 = pd.read_pickle("cache/w2v2.pkl")
label_df = pd.read_csv("EmoFilm_labels_v2.csv")

# create classifier and grouping object
clf = make_pipeline(
    StandardScaler(),
    SVC(gamma="auto"),
)

# Use 'speaker' as the grouping object for cross-validation
groups = label_df["speaker"]
cv = GroupKFold(n_splits=5)


# Define a function that returns the cross-validation score
def get_cv_score(clf, X, y, groups, cv):
    return cross_val_score(
        clf, X, y, groups=groups, cv=cv, scoring="balanced_accuracy")


# Loop over each C value to find the best one
best_score = 0
best_C = 0
for C in np.arange(0.1, 100.0, 0.1):
    # show progress
    print(f"Trying C={C}")
    clf.set_params(svc__C=C)
    scores = get_cv_score(clf, features_w2v2, label_df["emo"], groups, cv)
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_C = C

# Print the best C value and its corresponding score
print("Best C value:", best_C)
print("Best score:", best_score)
