#KNN cross validation with variable fold 

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

ds = "knntest.csv"
data = pd.read_csv(ds)
X = data.drop("win_superbowl", axis=1)
y = data["win_superbowl"]

# df = pd.read_csv('Preprocessed_Data.csv')
# db = df.values.tolist()  # all numeric now

db = np.array(data)
num_folds = 32

# Shuffle before splitting into folds
np.random.shuffle(db)

# Split into 32 roughly equal folds
folds = np.array_split(db, num_folds)

correct = 0
tp = 0
total = len(db)
tp = 0
tn = 0
fp = 0
fn = 0

# K-Fold Cross Validation (32 folds)
for fold in folds:
    # Test set = this fold
    X_test = [row[:-1] for row in fold]
    y_test = [row[-1] for row in fold]

    # Training set = all other folds combined
    train_data = np.vstack([f for f in folds if not np.array_equal(f, fold)])
    X_train = [row[:-1] for row in train_data]
    y_train = [row[-1] for row in train_data]

    smote = SMOTE(sampling_strategy='minority')
    X_res, Y_res = smote.fit_resample(X_train, y_train)
    #print("After SMOTE:\n", pd.Series(Y_res).value_counts())

    # Train classifier with k = 1
    clf = KNeighborsClassifier(n_neighbors=19, metric='euclidean')
    clf.fit(X_res, Y_res)

    # Predict all samples in this fold
    predictions = clf.predict(X_test)

    # Count correct
    for t, p in zip(y_test, predictions):
        if t == p:
            correct += 1
            if t == 1:
                tp += 1
            else:
                tn += 1
        elif t == 1:
            fp += 1
        elif t == 0:
            fn += 1
            

# Final accuracy and error rate
accuracy = correct / total
error_rate = 1 - accuracy
precison = tp / float(tp+fp)
recall = tp / float(tp+fn)
f1 = 2 / ((1/precison) + (1/recall))

print(tp)
print(correct)

print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print(f"True Positive: {tp}")
print(f"True Negative: {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"Precison: {precison}")
print(f"Recall: {recall}")
print(f"F1: {f1}")