import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# load data
features_w2v2 = pd.read_pickle("cache/w2v2.pkl")
label_df = pd.read_csv("EmoFilm_labels_v2.csv")

# create classifier and grouping object
clf = make_pipeline(
    StandardScaler(),
    SVC(C=3.4, kernel="rbf", gamma="auto"), # C=1.3; 3.9; 50.9; 2.6
)

# Use 'speaker' as the grouping object for cross-validation
groups = label_df["speaker"]

# Create a GroupKFold object
gkf = GroupKFold(n_splits=5)

# Initialize a list to store the accuracy scores
ua_scores = []
wa_scores = []

# Loop over each fold
for train_index, test_index in gkf.split(features_w2v2, label_df["emo"], groups):
    # Get the file names for training and test sets
    train_files = label_df.iloc[train_index]["file"]
    test_files = label_df.iloc[test_index]["file"]

    # Filter the features by file names
    train_features = features_w2v2.loc[train_files]
    test_features = features_w2v2.loc[test_files]

    # Filter the labels by file names
    train_labels = label_df.loc[train_files.index, "emo"]
    test_labels = label_df.loc[test_files.index, "emo"]

    # Fit the SVM classifier on the training data
    clf.fit(train_features, train_labels)

    # Use the SVM classifier to predict the test labels
    pred_labels = clf.predict(test_features)

    # Calculate the accuracy of the predictions
    ua = accuracy_score(test_labels, pred_labels)

    # Calculate the balanced accuracy of the predictions
    wa = balanced_accuracy_score(test_labels, pred_labels)

    # Print the unbalanced and balanced accuracy scores
    print("Fold", len(ua_scores) + 1, "- Unbalanced accuracy:", ua)
    print("Fold", len(wa_scores) + 1, "- Balanced accuracy:", wa)

    # Add the accuracy scores to the list
    ua_scores.append(ua)
    wa_scores.append(wa)

# Print the average accuracy scores over all folds
print("Average unbalanced accuracy:", sum(ua_scores) / len(ua_scores))
print("Average balanced accuracy:", sum(wa_scores) / len(wa_scores))
