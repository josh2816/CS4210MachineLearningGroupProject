#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB, CategoricalNB
import pandas as pd
from imblearn.over_sampling import SMOTE


dbTraining = []
dbTest = []


#Reading the training data using Pandas
df = pd.read_csv('training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

ds ='training.csv'
data = pd.read_csv('training.csv')
X = data.drop("win_superbowl", axis=1)
y = data["win_superbowl"]
count_class = y.value_counts()

smote = SMOTE(sampling_strategy='minority')
X_res, Y_res = smote.fit_resample(X, y)

# Convert to pandas if needed (SMOTE returns numpy arrays)
if not isinstance(X_res, pd.DataFrame):
    X_res = pd.DataFrame(X_res, columns=X.columns)
if not isinstance(Y_res, pd.Series):
    Y_res = pd.Series(Y_res)

print("After SMOTE:\n", Y_res.value_counts())

featureColumns = X_res.columns

mappings = {}
for col in featureColumns:
    categories = X_res[col].unique().tolist()
    # CategoricalNB requires non-negative integers starting from 0
    mappings[col] = {cat: i for i, cat in enumerate(categories)}  # Start from 0, not 1
    X_res[col] = X_res[col].map(mappings[col])

X = X_res[featureColumns].values
#print(X)

classCategories = Y_res.unique().tolist()
classMap = {cat: i+1 for i, cat in enumerate(classCategories)}
Y_res = Y_res.map(classMap)

Y = Y_res.values

clf = CategoricalNB()
clf = clf.fit(X, Y) #.predict(X_test)


dbTest = []
df_test = pd.read_csv('test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())


testX = []
for row in dbTest:
    test_features = []
    for j, col in enumerate(featureColumns):
        # CategoricalNB requires non-negative integers, use 0 as default instead of -1
        # If value not found, use the first category (index 0)
        default_value = 0 if len(mappings[col]) > 0 else 0
        test_features.append(mappings[col].get(row[j], default_value))
    testX.append(test_features)


#Printing the header of the solution
#--> add your Python code here
headers = df_test.columns.tolist()
for name in headers:
   print(f'{name:<8}', end = "")
print(" ", end = "")
print('Confidence', end="  ")
print('Predicted Class')

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
# Create reverse mapping for class labels (maps from model classes 1,2 back to original 0,1)
reverseClassMap = {v: k for k, v in classMap.items()}

# Map model class indices to original class values (0 or 1)
# sklearn's predict_proba returns probabilities in sorted order of clf.classes_
# clf.classes_ will be [1, 2] (sorted), so prob[0] = class 1, prob[1] = class 2
original_class_0 = reverseClassMap.get(1)  # Original class that was mapped to 1
original_class_1 = reverseClassMap.get(2)  # Original class that was mapped to 2

# Optional: create readable labels (0 = "No", 1 = "Yes")
label_map = {0: "No", 1: "Yes"}

correct = 0
tp = 0
tn = 0
fp = 0
fn = 0
total = 0
count = 0

for i in testX:
   prob = clf.predict_proba([i])[0]
   #print(prob)
   # Get the actual class from test data (last column in the row)
   actual_class = dbTest[count][-1]  # This is 0 or 1 from the test CSV
   
   if(prob[0] >= 0.7):
      p = 0
      total += 1
      row = df_test.loc[count]
      predicted_class = original_class_0  # This is 0 or 1
      for val in row:
        if val == '0.0':
            # Print the predicted class value (0 or 1)
            print(f'{predicted_class:<8}', end = "")
        else:
            print(f'{val:<11}', end = "")
      print(f'{round(prob[0], 3):<11}', end="  ")
      print(predicted_class)  # Print predicted class instead of actual
      if p == actual_class:
        correct += 1
        tn += 1
      else:
        fn += 1
    
   elif(prob[1] >= 0.7):
      p = 1
      total += 1
      row = df_test.loc[count]
      predicted_class = original_class_1  # This is 0 or 1
      for val in row:
        if val == '1.0':
            # Print the predicted class value (0 or 1)
            print(f'{predicted_class:<8}', end = "")
        else:
            print(f'{val:<11}', end = "")
      print(f'{round(prob[1], 3):<11}', end="  ")
      print(predicted_class)  # Print predicted class instead of actual
      if p == actual_class:
        correct += 1
        tp += 1
      else:
        fp += 1
    
   count+= 1

accuracy = correct/total
precison = tp / float(tp+fp)
recall = tp / float(tp+fn)
f1 = 2 / ((1/precison) + (1/recall))
print(f"Accuracy: {accuracy}" )
print(f"True Positive: {tp}")
print(f"True Negative: {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"Precison: {precison}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
