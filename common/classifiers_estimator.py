import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def estimate_classifiers(X_train, y_train, X_test, y_test):
    abc = []
    classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression', 'KNN', 'Decision Tree']
    models = [SVC(kernel='linear'), SVC(kernel='rbf'), LogisticRegression(), KNeighborsClassifier(n_neighbors=3),
              DecisionTreeClassifier()]
    for i in models:
        model = i
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        abc.append(metrics.accuracy_score(prediction, y_test))
    models_dataframe = pd.DataFrame(abc, index=classifiers)
    models_dataframe.columns = ['Accuracy']
    print(models_dataframe)
