import pandas as pd
import numpy as nm
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import linear_model, preprocessing

data = pd.read_csv("./Car Data Set/car.data")
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

names = ["unacc", "acc", "good", "vgood" ]
print(data.head())
print(buying)


X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# taking too high a value for k would mess up the algorithm
# for example, if k is 9, and there are 4 instance of the red group very close to our data point
# that we want to classify, and there are 5 instances of the blue group a bit further away,
# k's nearest neighbour will classify our point as blue even though it should be red

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predictions = model.predict(x_test)
for i in range(30):
    print(names[y_test[i]], names[predictions[i]], "\n")