#KNN leave one out

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('Preprocessed_Data.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

help = 0
total = 672
correct = 0
#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 2D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    X = [ [float(val) for val in row[:-1]] for row in db if row != i ]  


    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = [ row[-1] for row in db if row != i ]


    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    test_instance = [float(val) for val in i[:-1]]
    test_label = i[-1]

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X, Y)


    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([test_instance])[0]
    #print(class_predicted)
    if class_predicted == 1:
        help+=1

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if(test_label == class_predicted):
        correct += 1
        
    #total += 1

#Print the error rate
#--> add your Python code here
accuracy = correct/total
error = 1 - accuracy
print(f"Accuracy rate: {accuracy}")
print(f"Error rate: {error}")

print(help)






