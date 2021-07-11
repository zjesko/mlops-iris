from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# define a Gaussain NB classifier
gaussian = GaussianNB()
# define decision tree
Log_reg = LogisticRegression(random_state=10)
# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}
#define a classifier to store 
clf = gaussian
# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model (guassian)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    gaussian.fit(X_train, y_train)

    # do the test-train split and train the model (Decision Tree)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=12)
    Log_reg.fit(X_train, y_train)


    # calculate the print the accuracy score (gaussian)
    gaussian_acc = accuracy_score(y_test, gaussian.predict(X_test))
    print(f"Model trained with accuracy for guassian: {round(gaussian_acc, 3)}")

    # calculate the print the accuracy score (Decsion Tree)
    log_acc = accuracy_score(y_test, Log_reg.predict(X_test))
    print(f"Model trained with accuracy for logistic: {round(log_acc, 3)}")

    #check for the two classifier performance and return the best
    if gaussian_acc > log_acc :
        clf = gaussian_acc
        # print(clf)
    else: 
        clf = log_acc
        #print(clf)

# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
