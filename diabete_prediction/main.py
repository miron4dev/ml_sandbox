import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.svm import SVC

df = pd.read_csv('diabetes.csv')

# The data is clean
assert df.isnull().sum().all() == 0

features = df.drop('Outcome', axis=1)
target = df['Outcome']

steps = [('imputation', Imputer(missing_values=0, strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('SVM', SVC(C=1.0, gamma=0.01))]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=42)

# Fit to the training set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = pipeline.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(pipeline.score(X_test, y_test)))
print(classification_report(y_test, y_pred))

# Find the important features
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(features, target)
classifier = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
print(classifier)
