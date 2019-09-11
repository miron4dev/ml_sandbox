import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def estimate_classifiers(X_train, y_train, X_test, y_test):
    abc = []
    classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest',
                   'Naive Bayes', 'SGD']
    models = [SVC(kernel='linear'), SVC(kernel='rbf'), LogisticRegression(), KNeighborsClassifier(n_neighbors=3),
              DecisionTreeClassifier(), RandomForestClassifier(), MultinomialNB(), SGDClassifier(max_iter=1000, loss='log')]
    for i in models:
        model = i
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        abc.append(metrics.accuracy_score(prediction, y_test))
    models_dataframe = pd.DataFrame(abc, index=classifiers)
    models_dataframe.columns = ['Accuracy']
    print(models_dataframe)


def estimate_knn(X_train, y_train, X_test, y_test):
    a_index = list(range(1, 11))
    a = pd.Series()
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in list(range(1, 11)):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        a = a.append(pd.Series(metrics.accuracy_score(prediction, y_test)))
    plt.plot(a_index, a)
    plt.xticks(x)
    plt.show()
    print('Accuracies for different values of n are:', a.values)


def forest_classifier(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    print(pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False))
