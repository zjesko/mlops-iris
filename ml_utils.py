from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime

# define a Gaussain NB classifier
clf1 = GaussianNB()

# Task 3: Adding SVC classifer, train and compare accuracies between two fit the one with better accuracy
clf2 = SVC(kernel='poly', degree=3, max_iter=300000)

# variable to store model with higher accuracy
high_accracy = ""

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train the model 1
    clf1.fit(X_train, y_train)

    # calculate the print the accuracy score for model 1
    acc = accuracy_score(y_test, clf1.predict(X_test))
    print(f"Model trained with accuracy for : { clf1, round(acc, 3)}")

    # train the model 2
    clf2.fit(X_train, y_train)

    # calculate the print the accuracy score for model 2
    acc2 = accuracy_score(y_test, clf2.predict(X_test))
    print(f"Model trained with accuracy: {clf2, round(acc2, 3)}")

    # determine the better model, based on accuracy
    if acc >= acc2:
        high_accracy = "clf1"
    else:
        high_accracy = "clf2"
    print(acc, acc2, high_accracy)

# function to predict the flower using the model with better accuracy
def predict(query_data):
    x = list(query_data.dict().values())
    if high_accracy == "clf1":
        prediction = clf1.predict([x])[0]
    else:
        prediction = clf2.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again, based on the new data obtained
    if high_accracy == "clf1":
        clf1.fit(X, y)
    else:
        clf2.fit(X, y)
