import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

quadraic_featureizer = PolynomialFeatures(degree=2)
X_train_quad = quadraic_featureizer.fit_transform(X_train)
X_test_quad = quadraic_featureizer.fit_transform(X_test)

regressor_quad = LinearRegression()
regressor_quad.fit(X_train_quad, y_train)
xx_quad = quadraic_featureizer.transform(xx.reshape(xx.shape[0], 1))

plt.plot(xx, regressor_quad.predict(xx_quad), c='r', linestyle='-.')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()

print(X_train)
print(X_train_quad)
print(X_test)
print(X_test_quad)
print('Simple linear regression r-squared', regressor.score(X_test, y_test))
print('Quadratic regression r-squared', regressor_quad.score(X_test_quad, y_test))
