import tensorflow
import keras

import pandas as pd
import numpy as num
import sklearn
from sklearn import linear_model
import pickle
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

data =data[["G1", "G2", "G3", "studytime", "failures", "sex"]]
data["sex"] = data["sex"].map({"M": 1, "F": 0})

predict = "G3"

X = num.array(data.drop([predict], axis=1))
y = num.array(data[predict])
best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# for _ in range(30):
#
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#     # while training models from a pre-existing data set, we train the model from 90% of the data and withheld
#     # around 10% from it to test the model, since the model hasn't seen this data yet, we can compare the predictions
#     # it comes up with the actual values
#     # if we didn't do this, the model wil have access to 100% of the data and just give you the correct value since it
#     # has already seen those values before
#
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#
#     if acc > best:
#         best = acc
#         with open("studentModel.pickle", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("studentModel.pickle", "rb")
linear = pickle.load(pickle_in)
print(linear.score(x_test, y_test))
print(linear.coef_)  # the number of co-efficients would be equal to the number of variables you have
print(linear.intercept_)  # this tells us what the value would be if all coeffs were zero

predictions = linear.predict(x_test)
for i in range(30):
    print(int(predictions[i]), y_test[i])
