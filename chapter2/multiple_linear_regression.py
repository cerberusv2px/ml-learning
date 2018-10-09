from numpy.linalg import inv, lstsq
from sklearn.linear_model import LinearRegression
import numpy as np

# X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
# y = [[7], [9], [13], [17.5], [18]]
# beta = np.dot(inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
# print(beta)
# print(lstsq(X, y)[0])

X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]

model = LinearRegression()
model.fit(X, y)

X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]

predictions = model.predict(X_test)

for i, prediction in enumerate(predictions):
    print("Predicted: %s, Target: %s" % (prediction, y_test[i]))

print('R-squared: %.2f' % model.score(X_test, y_test))
