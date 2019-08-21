"""Predict the onset of diabetes based on diagnostic measures"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('diabetes.csv')

# The data is clean
assert df.isnull().sum().all() == 0

features = df.drop('Outcome', axis=1)
target = df['Outcome']

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Find the important features
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(features, target)
classifier = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
print(classifier)
