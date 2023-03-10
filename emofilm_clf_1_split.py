# classifier for fixed split
import pandas as pd
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

svm_pipeline = make_pipeline(
    StandardScaler(),
    SVC(C=1.3, kernel="rbf", gamma="scale"),
)


def svm_experiment(train_features, train_labels, test_features, test_labels):
    # Create an SVM classifier with default hyperparameters
    # svm = SVC(C=50.9, gamma="auto")

    # Fit the SVM classifier on the training data
    svm_pipeline.fit(train_features, train_labels)

    # Use the SVM classifier to predict the test labels
    pred_labels = svm_pipeline.predict(test_features)

    # Calculate the accuracy of the predictions
    ua = accuracy_score(test_labels, pred_labels)

    # Calculate the balanced accuracy of the predictions
    wa = balanced_accuracy_score(test_labels, pred_labels)

    # Print the unbalanced and balanced accuracy scores
    print("Unbalanced accuracy:", ua)
    print("Balanced accuracy:", wa)

    # Return the accuracy score
    return ua, wa


svm_experiment(train_features, train_labels, test_features, test_labels)

# output:
# (aud-vad-model) bagus@pc-omen:emofilm-vad-svm$ python emofilm_vad_svm_opt.py 
# Unbalanced accuracy: 0.7845303867403315
# Balanced accuracy: 0.7544329065908013