import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Loading the dataset
Diabetes_data = pd.read_csv("D:\Pythonn\Py_codes\ML\Diabetes_predict\diabetes.csv",header=0)

# Printing first 5 rows of dataset
Diabetes_data.head() 

# Number of rows and columns in the dataset
shape = Diabetes_data.shape
# print(shape)

# getting statistical measures of the data
desc = Diabetes_data.describe()
# print(desc)

val = Diabetes_data['Outcome'].value_counts()
# print(val)

m = Diabetes_data.groupby('Outcome').mean()
# print(m)

X = Diabetes_data.drop(columns='Outcome',axis=1)
Y = Diabetes_data['Outcome']
# print("-"*80)
# print(X)
# print("-"*80)
# print(Y)
# print("-"*80)

# Data Standardization
scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)
# print(standardized_data)

X = standardized_data
# print(X)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, stratify=Y, random_state= 2)
# print(X.shape,X_train.shape,X_test.shape)

# Training the model
Classifier = svm.SVC(kernel = 'linear')

# training the support vector machine(svm) classifier
datafit = Classifier.fit(X_train,Y_train)


# Accuracy score on the training data
X_train_prediction = Classifier.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f"Accuracy score of training data {training_accuracy}")


# Accuracy score on the test data
X_test_prediction = Classifier.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Accuracy score of testing data {testing_accuracy}")


input_data = (15,136,70,32,110,37.1,0.153,43)

# changing input_data to numpy array
input_arr = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_reshape = (input_arr.reshape(1,-1))
# print(input_reshape)

# standardize the input data
std_data = scaler.transform(input_reshape)
print(std_data)

prediction = Classifier.predict(std_data)
print(prediction)

if(prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")