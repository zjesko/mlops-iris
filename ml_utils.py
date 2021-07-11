from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# define a Gaussain NB classifier
gnb = GaussianNB()

# task 3
# define a Random Forest Classifier
rfc = RandomForestClassifier(max_depth=4, random_state=0)

best_clf = gnb

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    gnb.fit(X_train, y_train)

    # task 3 - training second classifier
    rfc.fit(X_train, y_train)

    # calculate the print the accuracy score of Gaussain NB
    acc_gnb = accuracy_score(y_test, gnb.predict(X_test))
    print(f"Gaussain NB Model trained with accuracy: {round(acc_gnb, 3)}")

    # task 3 - calculate the print the accuracy score of Random Forest Classifier
    acc_rfc = accuracy_score(y_test, rfc.predict(X_test))
    print(f"Random Forest Classifier Model trained with accuracy: {round(acc_rfc, 3)}")

    if acc_rfc > acc_gnb:
        best_clf = rfc
        print("Random Forest Classifier has better accuracy than Gaussain NB")
    else:
        best_clf = gnb
        print("Gaussain NB has better accuracy than Random Forest Classifier")


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = best_clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    best_clf.fit(X, y)
