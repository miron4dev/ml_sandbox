import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from common.accuracy_calculator import calculate_accuracy

train_values = pd.read_csv('train_values.csv')
train_labels = pd.read_csv('train_labels.csv')

le = LabelEncoder()


def optimize_data_set(data_frame):
    data_frame = data_frame.set_index('patient_id')
    data_frame['thal'] = le.fit_transform(data_frame['thal'])
    return data_frame


df = optimize_data_set(pd.merge(train_values, train_labels))

features = df.drop('heart_disease_present', axis=1)
target = df['heart_disease_present']

# Create the pipeline: pipeline
pipeline = Pipeline([
    ('imputation', SimpleImputer(missing_values=0, strategy='mean')),
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(C=1.0, penalty='l1', solver='liblinear'))]
)

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

# Fit to the training set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Compute and print metrics
calculate_accuracy(y_test, y_pred)
print('Loss Function: \n' + str(log_loss(y_test, y_pred_proba)))

test_values = optimize_data_set(pd.read_csv('test_values.csv'))

prediction = pipeline.predict_proba(test_values)[:, 1]
test_values['heart_disease_present'] = prediction
submission = test_values[['heart_disease_present']]
submission.to_csv('submission.csv', index=True)
