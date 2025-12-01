#ID3 test with smote


from sklearn import tree
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

ds ='training.csv'
data = pd.read_csv('training.csv')
X = data.drop("win_superbowl", axis=1)
y = data["win_superbowl"]

dbTest = []
df_test = pd.read_csv('test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

count_class = y.value_counts()
# plt.bar(count_class.index, count_class.values)
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.title('Class Distribution (Before SMOTE)')
# plt.xticks(count_class.index, ['Class 0', 'Class 1'])
# plt.show()

smote = SMOTE(sampling_strategy='minority')
X_res, Y_res = smote.fit_resample(X, y)
print("After SMOTE:\n", Y_res.value_counts())



featureColumns = X_res.columns

mappings = {}
for col in featureColumns:
    categories = X_res[col].unique().tolist()
    mappings[col] = {cat: i+1 for i, cat in enumerate(categories)}
    X_res[col] = X_res[col].map(mappings[col])

X = X_res[featureColumns].values
#print(X)

classCategories = Y_res.unique().tolist()
classMap = {cat: i+1 for i, cat in enumerate(classCategories)}
Y_res = Y_res.map(classMap)

Y = Y_res.values
#print(Y)



accuracy = 0
#Loop your training and test tasks 10 times here

# fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
# --> addd your Python code here
# clf =
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=12)
clf = clf.fit(X, Y)

#Read the test data and add this data to dbTest
#--> add your Python code here

correct = 0
total = 0
tp = 0
tn = 0
fp = 0
fn = 0
for data in dbTest:

    test_features = []
    for j, col in enumerate(featureColumns):
        test_features.append(mappings[col].get(data[j], -1))
    
    class_predicted = clf.predict([test_features])[0]
    

    #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
    #--> add your Python code here
    true_label = classMap[data[-1]]
    if(class_predicted == true_label):
        correct += 1
        if class_predicted == 2:
            tp += 1
        elif class_predicted == 1:
            tn += 1
    else:
        if class_predicted == 2:
            fp += 1
        elif class_predicted == 1:
            fn += 1

    total += 1

accuracy = float(correct/total)
precison = tp / float(tp+fp)
recall = tp / float(tp+fn)
f1 = 2 / ((1/precison) + (1/recall))

#Find the average of this model during the 10 runs (training and test set)
#--> add your Python code here

#Print the average accuracy of this model during the 10 runs (training and test set).
#Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
#--> add your Python code here

print(f"final accuracy when training on {ds}: {accuracy}" )
print(f"True Positive: {tp}")
print(f"True Negative: {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"Precison: {precison}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

