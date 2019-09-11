from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import uniform as sp_rand


def tune_linear_regression(X_train, y_train):
    param_grid = {
        'estimator__n_jobs': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
    tune_hyper_parameters(LinearRegression(), param_grid, X_train, y_train)


def tune_logistic_regression(X_train, y_train):
    param_grid = {
        'estimator__penalty': ['l1', 'l2'],
        'estimator__class_weight': ['balanced', None],
        'estimator__C': sp_rand(),
    }
    tune_hyper_parameters(LogisticRegression(solver='liblinear'), param_grid, X_train, y_train)


def tune_linear_svm(X_train, y_train):
    param_grid = {
        'estimator__C': [1, 10, 100, 1000],
        'estimator__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0],
    }
    tune_hyper_parameters(SVC(kernel='linear'), param_grid, X_train, y_train)


def tune_radial_svm(X_train, y_train):
    param_grid = {
        'estimator__C': [1, 10, 100, 1000],
        'estimator__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0],
    }
    tune_hyper_parameters(SVC(kernel='rbf'), param_grid, X_train, y_train)


def tune_random_forest(X_train, y_train):
    param_grid = {
        'estimator__criterion': ['gini', 'entropy'],
        'estimator__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }
    tune_hyper_parameters(RandomForestClassifier(), param_grid, X_train, y_train)


def tune_hyper_parameters(estimator, param_grid, X_train, y_train):
    pipeline = Pipeline([
        ('imputation', SimpleImputer(missing_values=0, strategy='mean')),
        ('scaler', StandardScaler()),
        ('estimator', estimator)
    ])

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(pipeline, param_grid, cv=5, scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()
