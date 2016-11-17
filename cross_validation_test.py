from sklearn import cross_validation
import numpy as np
x = np.array([[1,2], [3,4], [5,6], [7,8], [2,3], [67,45]])
y = np.array([1,2,3,4,5,8])

cv = cross_validation.KFold(6, n_folds=2)
print(len(cv))

for train_index, test_index in cv:
    print('TRAIN: ', train_index, "TEST: ", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(x_train, x_test, y_train, y_test)