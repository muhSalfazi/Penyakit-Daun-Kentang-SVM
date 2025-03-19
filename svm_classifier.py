from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def svm_train_test(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2)
    
    clf_linear = SVC(kernel='linear')
    clf_linear.fit(X_train, y_train)
    print('Train accuracy:', clf_linear.score(X_train, y_train))
    print('Test accuracy:', clf_linear.score(X_test, y_test))

    clf_rbf = SVC(kernel='rbf')
    clf_rbf.fit(X_train, y_train)
    print('Train accuracy:', clf_rbf.score(X_train, y_train))
    print('Test accuracy:', clf_rbf.score(X_test, y_test))

    test_preds = clf_linear.predict(X_test)
    print('Test Predicting:', accuracy_score(y_test, test_preds))

    cm = confusion_matrix(y_test, test_preds)
    label_names = ['early blight', 'late blight', 'healthy']
    
    plt.figure(figsize=(6, 6))
    plt.title('Confusion matrix on test data')
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False, xticklabels=label_names, yticklabels=label_names, annot_kws={'size': 12})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
