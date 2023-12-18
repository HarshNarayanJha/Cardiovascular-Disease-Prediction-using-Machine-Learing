import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from dataset import load_cardio_dataset

from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = pd.read_csv("Cardiovascular Disease Prediction using Machine Learing/cardio_train.csv", delimiter=";")
print(dataset.head())

dataset["age"] = dataset["age"] / dataset["age"].abs().max()

# dataset["gender"] = dataset["gender"].map(lambda x: "Male" if x == 1 else "Female")
# dataset["smoke"] = dataset["smoke"].map(lambda x: "Yes" if x == 1 else "No")
# dataset["alco"] = dataset["alco"].map(lambda x: "Yes" if x == 1 else "No")
# dataset["active"] = dataset["active"].map(lambda x: "Yes" if x == 1 else "No")
# dataset["cardio"] = dataset["cardio"].map(lambda x: "Yes" if x == 1 else "No")
# dataset["cholesterol"] = dataset["cholesterol"].map(lambda x: {1: "Low", 2: "Med", 3: "High"}[x])
# dataset["gluc"] = dataset["gluc"].map(lambda x: {1: "Low", 2: "Med", 3: "High"}[x])
# print(dataset.columns)

renaming_map = {'id': "Id", 'age': "Age", 'gender': "Gender", 'height': "Height", 'weight': "Weight", 'ap_hi': "ap_hi", 'ap_lo': "ap_lo",
                'cholesterol': "Cholesterol", 'gluc': "Glucose", 'smoke': "Smokes", 'alco': "Alcoholic", 'active': "Active", 'cardio': "Cardio"}

# dataset.rename(renaming_map, inplace=True)
dataset.columns = renaming_map.values()

print(dataset.corr())

# plt.plot(dataset["Age"], dataset["Cholesterol"])
# plt.show()

print(dataset.nunique())

dataset = load_cardio_dataset()

# Support Vector Machine (SVM)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)

# clf = svm.SVC(kernel='linear')

clf = LinearSVC(random_state=0, tol=1e-5, dual=1.5)
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)

print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))

# K-Nearest Neighbour (KNN)

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Decision Trees (DT) 

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("DT Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Logistic Regression (LR)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)

logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print("LR Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Random Forest (RF)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=50)
rf = RandomForestClassifier()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("RF Accuracy:",metrics.accuracy_score(y_test, y_pred))


