import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors  import  KNeighborsClassifier
from sklearn import metrics

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

classes = ["malignant", "benign"]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# print(x_train, y_train)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
kAcc = model.score(x_test, y_test)

clf = svm.SVC(kernel="linear", C=3)
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, predictions)

print(acc, kAcc, "\n")
