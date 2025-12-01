#-------------------------------------------------------------------------
# AUTHOR: Kate Yuan
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
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
    mappings[col] = {cat: i+1 for i, cat in enumerate(categories)}
    X_res[col] = X_res[col].map(mappings[col])

X = X_res[featureColumns].values
#print(X)

classCategories = Y_res.unique().tolist()
classMap = {cat: i+1 for i, cat in enumerate(classCategories)}
Y_res = Y_res.map(classMap)

Y = Y_res.values
#print(Y)



#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X = []
# one = ['Sunny', 'Hot', 'High', 'Strong']
# two = ['Overcast', 'Mild', 'Normal', 'Weak']
# three =['Rain', 'Cool']
# rowLen = len(dbTraining[0])
# for i in range(len(dbTraining)):
#   X.append([])
#   for j in range(rowLen - 1):
#     if dbTraining[i][j] in one:
#         X[i].append(1)
#     elif dbTraining[i][j] in two:
#         X[i].append(2)
#     elif dbTraining[i][j] in three:
#         X[i].append(3)
# #print(X)

# #Transform the original training classes to numbers and add them to the vector Y.
# #For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# #--> add your Python code here
# Y = []
# values = ["Yes", "No"]
# rowLen = len(dbTraining[0])
# for i in range(len(dbTraining)):
#   if dbTraining[i][rowLen - 1] == values[0]:
#       Y.append(1)
#   elif dbTraining[i][rowLen - 1] == values[1]:
#       Y.append(2)

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf = clf.fit(X, Y) #.predict(X_test)


dbTest = []
df_test = pd.read_csv('test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())


testX = []
for row in dbTest:
    test_features = []
    for j, col in enumerate(featureColumns):
        test_features.append(mappings[col].get(row[j], -1))
    testX.append(test_features)


#Printing the header of the solution
#--> add your Python code here
headers = df_test.columns.tolist()
for name in headers:
   print(f'{name:<8}', end = "")
print(" ", end = "")
print('Confidence')

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

count = 0
for i in testX:
   prob = clf.predict_proba([i])[0]
   if(prob[0] > 0.30):
      row = df_test.loc[count]
      predicted_class = original_class_0  # This is 0 or 1
      for val in row:
        if val == '?':
            # Print the actual class value (0 or 1), or use label_map for readable output
            print(f'{predicted_class:<8}', end = "")
        else:
            print(f'{val:<11}', end = "")
      print(round(prob[0], 3))
   elif(prob[1] > 0.30):
      row = df_test.loc[count]
      predicted_class = original_class_1  # This is 0 or 1
      for val in row:
        if val == '?':
            # Print the actual class value (0 or 1), or use label_map for readable output
            print(f'{predicted_class:<8}', end = "")
        else:
            print(f'{val:<11}', end = "")
      print(round(prob[1], 3))
   count+= 1


