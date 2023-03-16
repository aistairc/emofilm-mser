# code to optimize C parameter in SVM classifier
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

# load data
features_w2v2 = pd.read_pickle("cache/w2v2.pkl")

# Read the CSV file
label_df = pd.read_csv("EmoFilm_labels_16k_split.csv")

# Extract file names for training and test sets
train_files = label_df.loc[label_df["set"] == 0, "file"]
test_files = label_df.loc[label_df["set"] == 1, "file"]

# Filter features_w2v2 by file names
train_features = features_w2v2.loc[train_files]
test_features = features_w2v2.loc[test_files]


# Filter emo labels by file names
train_labels = label_df.loc[train_files.index, "emo"]
test_labels = label_df.loc[test_files.index, "emo"]

list_ua = []
list_wa = []
opt_c = np.arange(0.1, 100, 0.1)


def svm_experiment(train_features, train_labels, test_features, test_labels,    opt_c):
    # Create an SVM classifier with default hyperparameters
    svm = SVC(C=opt_c, gamma="scale")

    # Fit the SVM classifier on the training data
    svm.fit(train_features, train_labels)

    # Use the SVM classifier to predict the test labels
    pred_labels = svm.predict(test_features)

    # Calculate the accuracy of the predictions
    ua = accuracy_score(test_labels, pred_labels)

    # Calculate the balanced accuracy of the predictions
    wa = balanced_accuracy_score(test_labels, pred_labels)

    # Print the unbalanced and balanced accuracy scores
    print("Unbalanced accuracy:", ua)
    print("Balanced accuracy:", wa)

    # Return the accuracy score
    list_ua.append(ua)
    list_wa.append(wa)
    return ua, wa


# Run the experiment with dfiferent C values
for c in opt_c:
    print("C:", c)
    svm_experiment(train_features, train_labels, test_features, test_labels, c)

# print max unbalanced accuracy
print(f"Max unbalanced accuracy: {max(list_ua)} C={opt_c[np.argmax(list_ua)]}")
# print max balanced accuracy
print(f"Max balanced accuracy: {max(list_wa)} C={opt_c[np.argmax(list_wa)]}")