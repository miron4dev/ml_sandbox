from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix


def calculate_accuracy(y_test, y_pred):
    print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
    print('Precision Score : ' + str(precision_score(y_test, y_pred)))
    print('Recall Score : ' + str(recall_score(y_test, y_pred)))
    print('F1 Score : ' + str(f1_score(y_test, y_pred)))

    print('Confusion Matrix : \n' + str(confusion_matrix(y_test, y_pred)))
