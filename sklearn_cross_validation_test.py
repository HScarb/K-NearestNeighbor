import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
iris = datasets.load_iris()

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.1, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
clf = RandomForestClassifier()
# train with randomForest
print(" training begin...")
clf = clf.fit(x_train,y_train)
print(" training end.")
#==========================================================================
# test randomForestClassifier with testsets
print(" test begin.")
predict_ = clf.predict(x_test) #return type is float64
proba = clf.predict_proba(x_test) #return type is float64
score_ = clf.score(x_test,y_test)
print(" test end.")

print(score_)